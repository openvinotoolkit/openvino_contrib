/*
 * Fused Sparse Convolution OpenCL Kernels
 * 
 * Key optimizations vs. the old gather-GEMM-scatter approach:
 * 1. Neighbor map computed on GPU (no CPU hash table)
 * 2. Single fused kernel per conv layer (no 27 separate GPU dispatches)
 * 3. All data stays on GPU between layers (no CPU readback)
 * 4. Tiled accumulation over C_in for better cache behavior
 * 5. Fused BN+ReLU in the same kernel
 *
 * For submanifold convolution: output indices == input indices
 * For strided convolution: uses separate output coordinate generation
 */

// ============================================================
// GPU Hash Table: Build coord→index lookup
// Fills a dense hash table [X*Y*Z] = -1 initially, 
// then sets hash_table[x*Y*Z + y*Z + z] = voxel_index
// ============================================================
__kernel void build_hash_table(
    __global const int* coords,       // [N, 4]: batch, x, y, z
    __global int* hash_table,         // [X*Y*Z]: -1 = empty
    const int N,
    const int Y,
    const int Z
) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];
    
    int hash_idx = (x * Y + y) * Z + z;
    hash_table[hash_idx] = idx;
}

// ============================================================
// Compact Open-Addressing GPU Hash Table
// Uses power-of-2 sized table with linear probing.
// Much smaller than dense table for sparse grids.
// ============================================================
__kernel void build_hash_table_compact(
    __global const int* coords,       // [N, 4]: batch, x, y, z
    __global int* hash_keys,          // [TABLE_SIZE]: keys (-1 = empty)
    __global int* hash_vals,          // [TABLE_SIZE]: values
    const int N,
    const int Y,
    const int Z,
    const int TABLE_MASK              // TABLE_SIZE - 1
) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];
    
    int key = (x * Y + y) * Z + z;
    uint h = ((uint)key * 2654435761u) & (uint)TABLE_MASK;
    
    // Linear probing insert with atomic CAS
    while (true) {
        int old = atomic_cmpxchg(&hash_keys[h], -1, key);
        if (old == -1 || old == key) {
            hash_vals[h] = idx;
            return;
        }
        h = (h + 1) & (uint)TABLE_MASK;
    }
}

// ============================================================
// GPU Neighbor Map using compact hash table
// ============================================================
__kernel void build_neighbor_map_compact(
    __global const int* coords,       // [N, 4]: batch, x, y, z
    __global const int* hash_keys,    // [TABLE_SIZE]: keys
    __global const int* hash_vals,    // [TABLE_SIZE]: values
    __global int* neighbor_map,       // [N, 27]: output
    const int N,
    const int X,
    const int Y,
    const int Z,
    const int TABLE_MASK
) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    int bx = coords[idx * 4 + 1];
    int by = coords[idx * 4 + 2];
    int bz = coords[idx * 4 + 3];
    
    int k = 0;
    for (int ddx = -1; ddx <= 1; ddx++) {
        for (int ddy = -1; ddy <= 1; ddy++) {
            for (int ddz = -1; ddz <= 1; ddz++) {
                int nx = bx + ddx;
                int ny = by + ddy;
                int nz = bz + ddz;
                int result = -1;
                
                if (nx >= 0 && nx < X && ny >= 0 && ny < Y && nz >= 0 && nz < Z) {
                    int key = (nx * Y + ny) * Z + nz;
                    uint h = ((uint)key * 2654435761u) & (uint)TABLE_MASK;
                    while (true) {
                        int k_stored = hash_keys[h];
                        if (k_stored == -1) break;
                        if (k_stored == key) { result = hash_vals[h]; break; }
                        h = (h + 1) & (uint)TABLE_MASK;
                    }
                }
                neighbor_map[idx * 27 + k] = result;
                k++;
            }
        }
    }
}

// ============================================================
// GPU Neighbor Map: For each voxel, find 27 neighbors
// neighbor_map[voxel * 27 + k] = index of neighbor or -1
// ============================================================
__kernel void build_neighbor_map_kernel(
    __global const int* coords,       // [N, 4]: batch, x, y, z
    __global const int* hash_table,   // [X*Y*Z]: coord -> index
    __global int* neighbor_map,       // [N, 27]: output
    const int N,
    const int X,
    const int Y,
    const int Z
) {
    int idx = get_global_id(0);
    if (idx >= N) return;
    
    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];
    
    int k = 0;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                
                if (nx >= 0 && nx < X && ny >= 0 && ny < Y && nz >= 0 && nz < Z) {
                    int hash_idx = (nx * Y + ny) * Z + nz;
                    neighbor_map[idx * 27 + k] = hash_table[hash_idx];
                } else {
                    neighbor_map[idx * 27 + k] = -1;
                }
                k++;
            }
        }
    }
}

// ============================================================
// Fused Submanifold Sparse Convolution
// 
// Each work-item computes one (voxel, c_out) pair.
// Loops over 27 neighbors and C_in channels internally.
// Includes fused BN + optional ReLU.
//
// Weight layout: [27, C_in, C_out] (pre-reformatted)
// ============================================================
__kernel void subm_conv_fused(
    __global const float* input_features,    // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const float* weights,           // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global float* output_features,         // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int voxel_idx = get_global_id(0);
    int c_out = get_global_id(1);
    
    if (voxel_idx >= N || c_out >= C_out) return;
    
    float sum = 0.0f;
    
    // Loop over 27 neighbor positions
    for (int k = 0; k < 27; k++) {
        int neighbor_idx = neighbor_map[voxel_idx * 27 + k];
        if (neighbor_idx < 0) continue;
        
        // Dot product over input channels
        // weight index: weights[(k * C_in + c_in) * C_out + c_out]
        float partial = 0.0f;
        __global const float* in_row = input_features + neighbor_idx * C_in;
        __global const float* w_row = weights + (k * C_in) * C_out + c_out;
        
        for (int c_in = 0; c_in < C_in; c_in++) {
            partial += in_row[c_in] * w_row[c_in * C_out];
        }
        
        sum += partial;
    }
    
    // Fused BN + ReLU
    sum = sum * bn_scale[c_out] + bn_bias[c_out];
    if (apply_relu && sum < 0.0f) sum = 0.0f;
    
    output_features[voxel_idx * C_out + c_out] = sum;
}

// ============================================================
// Multi-output-channel variant: each work-item processes 4 c_out
// Better instruction-level parallelism and register reuse
// ============================================================
#define COUT_PER_WI 4

__kernel void subm_conv_fused_v2(
    __global const float* input_features,    // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const float* weights,           // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global float* output_features,         // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int voxel_idx = get_global_id(0);
    int c_out_base = get_global_id(1) * COUT_PER_WI;
    
    if (voxel_idx >= N || c_out_base >= C_out) return;
    
    float sums[COUT_PER_WI];
    for (int i = 0; i < COUT_PER_WI; i++) sums[i] = 0.0f;
    
    for (int k = 0; k < 27; k++) {
        int neighbor_idx = neighbor_map[voxel_idx * 27 + k];
        if (neighbor_idx < 0) continue;
        
        __global const float* in_row = input_features + neighbor_idx * C_in;
        
        for (int c_in = 0; c_in < C_in; c_in++) {
            float in_val = in_row[c_in];
            __global const float* w_base = weights + (k * C_in + c_in) * C_out + c_out_base;
            
            for (int i = 0; i < COUT_PER_WI; i++) {
                if (c_out_base + i < C_out) {
                    sums[i] += in_val * w_base[i];
                }
            }
        }
    }
    
    // Fused BN + ReLU and write
    for (int i = 0; i < COUT_PER_WI; i++) {
        int c_out = c_out_base + i;
        if (c_out < C_out) {
            float val = sums[i] * bn_scale[c_out] + bn_bias[c_out];
            if (apply_relu && val < 0.0f) val = 0.0f;
            output_features[voxel_idx * C_out + c_out] = val;
        }
    }
}


// ============================================================
// Tiled submanifold convolution: uses local memory to share
// neighbor features across all c_out work-items for the same voxel.
// Work-group = (C_out/4 threads) processing 1 voxel.
// Eliminates redundant global reads (up to 32x for C_out=128).
// ============================================================
__kernel void subm_conv_tiled(
    __global const float* input_features,    // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const float* weights,           // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global float* output_features,         // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local int* local_nmap,                 // [27]
    __local float* local_feat                // [27 * C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);
    
    if (voxel_idx >= N) return;
    
    // Cooperatively load neighbor map
    for (int k = lid; k < 27; k += WG) {
        local_nmap[k] = neighbor_map[voxel_idx * 27 + k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Cooperatively load features for valid neighbors
    for (int k = 0; k < 27; k++) {
        int nb = local_nmap[k];
        if (nb >= 0) {
            __global const float* row = input_features + nb * C_in;
            for (int c = lid; c < C_in; c += WG) {
                local_feat[k * C_in + c] = row[c];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int c_out_base = lid * 4;
    if (c_out_base >= C_out) return;
    
    // Compute convolution from local memory
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int k = 0; k < 27; k++) {
        if (local_nmap[k] < 0) continue;
        __local const float* f_row = local_feat + k * C_in;
        int w_base_k = k * C_in * C_out;
        for (int c = 0; c < C_in; c++) {
            float fv = f_row[c];
            int w_off = w_base_k + c * C_out + c_out_base;
            s0 += fv * weights[w_off];
            s1 += fv * weights[w_off + 1];
            s2 += fv * weights[w_off + 2];
            s3 += fv * weights[w_off + 3];
        }
    }

    // BN + ReLU + Store
    int base = voxel_idx * C_out + c_out_base;
    float v0 = s0 * bn_scale[c_out_base]     + bn_bias[c_out_base];
    float v1 = s1 * bn_scale[c_out_base + 1] + bn_bias[c_out_base + 1];
    float v2 = s2 * bn_scale[c_out_base + 2] + bn_bias[c_out_base + 2];
    float v3 = s3 * bn_scale[c_out_base + 3] + bn_bias[c_out_base + 3];
    if (apply_relu) {
        v0 = fmax(v0, 0.0f); v1 = fmax(v1, 0.0f);
        v2 = fmax(v2, 0.0f); v3 = fmax(v3, 0.0f);
    }
    output_features[base]     = v0;
    output_features[base + 1] = v1;
    output_features[base + 2] = v2;
    output_features[base + 3] = v3;
}


// ============================================================
// Residual Add + ReLU
// output = ReLU(a + b)
// ============================================================
__kernel void residual_add_relu(
    __global const float* a,       // [N, C]
    __global const float* b,       // [N, C]
    __global float* output,        // [N, C]
    const int total_elements
) {
    int idx = get_global_id(0);
    if (idx >= total_elements) return;
    
    float val = a[idx] + b[idx];
    output[idx] = val > 0.0f ? val : 0.0f;
}


// ============================================================
// Strided Sparse Convolution (for downsampling)
//
// Unlike SubM, output voxels are different from input voxels.
// Uses gather-GEMM approach but fused into single kernel dispatch.
//
// Approach: For each input voxel, compute which output positions it 
// contributes to, and atomically add the contribution.
// 
// This kernel processes one (input_voxel, c_out) pair.
// ============================================================
__kernel void strided_conv_scatter(
    __global const float* input_features,   // [N_in, C_in]
    __global const int* input_coords,       // [N_in, 4]: batch, x, y, z  
    __global const float* weights,          // [num_k, C_in, C_out]
    __global const float* bn_scale,         // [C_out]
    __global const float* bn_bias,          // [C_out]
    __global float* output_features,        // [N_out, C_out] - pre-zeroed
    __global const int* out_hash_table,     // [X_out * Y_out * Z_out] -> out voxel idx
    const int N_in,
    const int C_in,
    const int C_out,
    const int kx, const int ky, const int kz,
    const int sx, const int sy, const int sz,
    const int px, const int py, const int pz,
    const int X_out, const int Y_out, const int Z_out
) {
    int in_idx = get_global_id(0);
    int c_out = get_global_id(1);
    
    if (in_idx >= N_in || c_out >= C_out) return;
    
    int in_x = input_coords[in_idx * 4 + 1];
    int in_y = input_coords[in_idx * 4 + 2];
    int in_z = input_coords[in_idx * 4 + 3];
    
    // For each kernel position, check if this input maps to a valid output
    int k = 0;
    for (int ki = 0; ki < kx; ki++) {
        for (int kj = 0; kj < ky; kj++) {
            for (int kk = 0; kk < kz; kk++) {
                int ox_val = in_x + px - ki;
                int oy_val = in_y + py - kj;
                int oz_val = in_z + pz - kk;
                
                if (ox_val >= 0 && ox_val % sx == 0 &&
                    oy_val >= 0 && oy_val % sy == 0 &&
                    oz_val >= 0 && oz_val % sz == 0) {
                    
                    int out_x = ox_val / sx;
                    int out_y = oy_val / sy;
                    int out_z = oz_val / sz;
                    
                    if (out_x >= 0 && out_x < X_out &&
                        out_y >= 0 && out_y < Y_out &&
                        out_z >= 0 && out_z < Z_out) {
                        
                        int out_hash_idx = (out_x * Y_out + out_y) * Z_out + out_z;
                        int out_idx = out_hash_table[out_hash_idx];
                        
                        if (out_idx >= 0) {
                            // Compute dot product for this kernel position
                            float dot = 0.0f;
                            __global const float* in_row = input_features + in_idx * C_in;
                            __global const float* w_row = weights + (k * C_in) * C_out + c_out;
                            
                            for (int c_in = 0; c_in < C_in; c_in++) {
                                dot += in_row[c_in] * w_row[c_in * C_out];
                            }
                            
                            // Atomic add to output (multiple inputs may map to same output)
                            // Use CAS loop for float atomic add
                            __global float* addr = output_features + out_idx * C_out + c_out;
                            union { unsigned int u; float f; } old_val, new_val;
                            do {
                                old_val.f = *addr;
                                new_val.f = old_val.f + dot;
                            } while (atomic_cmpxchg((__global unsigned int*)addr, old_val.u, new_val.u) != old_val.u);
                        }
                    }
                }
                k++;
            }
        }
    }
}


// ============================================================
// Apply BN + ReLU to pre-accumulated strided conv output
// ============================================================  
__kernel void apply_bn_relu(
    __global float* features,        // [N, C] - modified in-place
    __global const float* bn_scale,  // [C]
    __global const float* bn_bias,   // [C]
    const int N,
    const int C,
    const int apply_relu
) {
    int idx = get_global_id(0);
    int c = get_global_id(1);
    
    if (idx >= N || c >= C) return;
    
    int i = idx * C + c;
    float val = features[i] * bn_scale[c] + bn_bias[c];
    if (apply_relu && val < 0.0f) val = 0.0f;
    features[i] = val;
}


// ============================================================
// Generate output coordinates for strided convolution
// Each input voxel checks all kernel positions to find valid outputs.
// output_flags[in_idx * num_k + k] = 1 if valid output
// ============================================================
__kernel void generate_strided_output_coords(
    __global const int* input_coords,   // [N_in, 4]: batch, x, y, z
    __global int* output_flags,         // [N_in * num_k]: 1 if valid output
    __global int* output_coords_flat,   // [N_in * num_k * 4]: output coords (batch, ox, oy, oz)
    const int N_in,
    const int kx, const int ky, const int kz,
    const int sx, const int sy, const int sz,
    const int px, const int py, const int pz,
    const int X_out, const int Y_out, const int Z_out
) {
    int in_idx = get_global_id(0);
    if (in_idx >= N_in) return;
    
    int batch = input_coords[in_idx * 4 + 0];
    int in_x = input_coords[in_idx * 4 + 1];
    int in_y = input_coords[in_idx * 4 + 2];
    int in_z = input_coords[in_idx * 4 + 3];
    
    int num_k = kx * ky * kz;
    int k = 0;
    for (int ki = 0; ki < kx; ki++) {
        for (int kj = 0; kj < ky; kj++) {
            for (int kk = 0; kk < kz; kk++) {
                int flat = in_idx * num_k + k;
                int ox_val = in_x + px - ki;
                int oy_val = in_y + py - kj;
                int oz_val = in_z + pz - kk;
                
                int valid = 0;
                int out_x = 0, out_y = 0, out_z = 0;
                
                if (ox_val >= 0 && ox_val % sx == 0 &&
                    oy_val >= 0 && oy_val % sy == 0 &&
                    oz_val >= 0 && oz_val % sz == 0) {
                    out_x = ox_val / sx;
                    out_y = oy_val / sy;
                    out_z = oz_val / sz;
                    
                    if (out_x >= 0 && out_x < X_out &&
                        out_y >= 0 && out_y < Y_out &&
                        out_z >= 0 && out_z < Z_out) {
                        valid = 1;
                    }
                }
                
                output_flags[flat] = valid;
                output_coords_flat[flat * 4 + 0] = batch;
                output_coords_flat[flat * 4 + 1] = out_x;
                output_coords_flat[flat * 4 + 2] = out_y;
                output_coords_flat[flat * 4 + 3] = out_z;
                
                k++;
            }
        }
    }
}


// ============================================================
// Sparse to Dense conversion (GPU version)
// ============================================================
__kernel void sparse_to_dense_kernel(
    __global const float* features,  // [N, C]
    __global const int* coords,      // [N, 4]: batch, x, y, z 
    __global float* dense,           // [B, C, X, Y, Z]
    const int N,
    const int C,
    const int X,
    const int Y,
    const int Z
) {
    int idx = get_global_id(0);
    int c = get_global_id(1);
    
    if (idx >= N || c >= C) return;
    
    int batch = coords[idx * 4 + 0];
    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];
    
    if (x >= 0 && x < X && y >= 0 && y < Y && z >= 0 && z < Z) {
        int dense_idx = ((((batch * C + c) * X + x) * Y + y) * Z + z);
        dense[dense_idx] = features[idx * C + c];
    }
}


// ============================================================
// FP16 Kernels - halve memory bandwidth, double throughput
// ============================================================
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Convert FP32 buffer to FP16
__kernel void float_to_half_kernel(
    __global const float* input,
    __global half* output,
    const int total
) {
    int idx = get_global_id(0);
    if (idx >= total) return;
    output[idx] = convert_half(input[idx]);
}

// Fused BN + ReLU + FP32→FP16: apply batch norm, optional relu,
// and convert to half precision in a single pass.
// Saves 2 kernel launches and ~2.3x memory traffic vs separate kernels.
__kernel void fused_bn_relu_f16(
    __global const float* features,     // [N, C] FP32 input
    __global const float* bn_scale,     // [C]
    __global const float* bn_bias,      // [C]
    __global half* output,              // [N, C] FP16 output
    const int N,
    const int C,
    const int apply_relu
) {
    int idx = get_global_id(0);
    int c = get_global_id(1);
    
    if (idx >= N || c >= C) return;
    
    int i = idx * C + c;
    float val = features[i] * bn_scale[c] + bn_bias[c];
    if (apply_relu && val < 0.0f) val = 0.0f;
    output[i] = convert_half(val);
}

// Convert FP16 buffer to FP32
__kernel void half_to_float_kernel(
    __global const half* input,
    __global float* output,
    const int total
) {
    int idx = get_global_id(0);
    if (idx >= total) return;
    output[idx] = convert_float(input[idx]);
}

// FP16 submanifold convolution: features and weights in FP16,
// accumulation in FP32 for precision, output in FP16
// Processes 8 output channels per work-item to amortize feature reads
//
// DIMENSION LAYOUT: gws = {Cout/8, N}
// dim0 = output channel group (SIMD lanes share same voxel → L1 broadcast)
// dim1 = voxel index
__kernel void subm_conv_fp16(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 8;  // dim0: output channels (SIMD-friendly)
    int voxel_idx = get_global_id(1);        // dim1: voxel index

    if (voxel_idx >= N || c_out_base >= C_out) return;

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            half8 w8 = vload8(0, weights + w_k_base + c * C_out);
            acc += fv * convert_float8(w8);
        }
    }

    // BN + ReLU in FP32 (vectorized)
    float8 sc = vload8(0, bn_scale + c_out_base);
    float8 bi = vload8(0, bn_bias + c_out_base);
    float8 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float8)(0.0f));
    }

    vstore8(convert_half8(v), 0, output_features + voxel_idx * C_out + c_out_base);
}

// ============================================================
// Scalar subm conv: 1 output channel per work-item
// Relies on hardware SIMD coalescing across output channels.
// Minimal register pressure → no register spilling.
// gws = {C_out, N}
// ============================================================
__kernel void subm_conv_fp16_scalar(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int co = get_global_id(0);               // output channel (coalesced across SIMD lanes)
    int vi = get_global_id(1);               // voxel index (shared across SIMD lanes)

    if (vi >= N || co >= C_out) return;

    float acc = 0.0f;

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[vi * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        __global const half* w_ptr = weights + (k * C_in + 0) * C_out + co;

        for (int c = 0; c < C_in; c++) {
            acc += convert_float(in_row[c]) * convert_float(w_ptr[c * C_out]);
        }
    }

    acc = acc * bn_scale[co] + bn_bias[co];
    if (apply_relu && acc < 0.0f) acc = 0.0f;
    output_features[vi * C_out + co] = convert_half(acc);
}

// ============================================================
// float4 subm conv: 4 output channels per work-item
// Moderate vectorization — balances register pressure and feature reuse.
// gws = {C_out/4, N}
// ============================================================
__kernel void subm_conv_fp16_w4(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 4;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            half4 w4 = vload4(0, weights + w_k_base + c * C_out);
            acc += fv * convert_float4(w4);
        }
    }

    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float4)(0.0f));
    }

    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + c_out_base);
}

// ============================================================
// Transposed-weight subm conv: 4 output channels per work item
// Weight layout: [27, C_out/4, C_in, 4] instead of [27, C_in, C_out]
// This gives 8-byte stride in inner loop vs 256-byte stride,
// improving cache line utilization from 12.5% to 100%.
// gws = {C_out/4, N}
// ============================================================
__kernel void subm_conv_fp16_w4t(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights_t,          // [27, C_out/4, C_in, 4] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int group = get_global_id(0);    // output channel group [0, C_out/4)
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || group * 4 >= C_out) return;

    float4 acc = (float4)(0.0f);
    int c_out_4 = C_out >> 2;

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        // Weight offset: k * (C_out/4 * C_in * 4) + group * (C_in * 4)
        __global const half* w_ptr = weights_t + (k * c_out_4 + group) * (C_in * 4);

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            half4 w4 = vload4(c, w_ptr);  // stride = 4 halfs = 8 bytes
            acc += fv * convert_float4(w4);
        }
    }

    int c_out_base = group * 4;
    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float4)(0.0f));
    }
    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + c_out_base);
}

// ============================================================
// Transposed-weight subm conv: 8 output channels per work item
// Weight layout: [27, C_out/8, C_in, 8] instead of [27, C_in, C_out]
// gws = {C_out/8, N}
// ============================================================
__kernel void subm_conv_fp16_w8t(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights_t,          // [27, C_out/8, C_in, 8] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int group = get_global_id(0);
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || group * 8 >= C_out) return;

    float8 acc = (float8)(0.0f);
    int c_out_8 = C_out >> 3;

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        __global const half* w_ptr = weights_t + (k * c_out_8 + group) * (C_in * 8);

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            half8 w8 = vload8(c, w_ptr);
            acc += fv * convert_float8(w8);
        }
    }

    int c_out_base = group * 8;
    float8 sc = vload8(0, bn_scale + c_out_base);
    float8 bi = vload8(0, bn_bias + c_out_base);
    float8 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float8)(0.0f));
    }
    vstore8(convert_half8(v), 0, output_features + voxel_idx * C_out + c_out_base);
}

// ============================================================
// Wide subm conv: 16 output channels per work item (FP16)
// Halves feature reads for large channel counts (C_out >= 32)
// gws = {C_out/16, N}
// ============================================================
__kernel void subm_conv_fp16_w16(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 16;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    float8 acc0 = (float8)(0.0f);
    float8 acc1 = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            int w_off = w_k_base + c * C_out;
            acc0 += fv * convert_float8(vload8(0, weights + w_off));
            acc1 += fv * convert_float8(vload8(1, weights + w_off));
        }
    }

    // BN + ReLU (vectorized)
    float8 v0 = acc0 * vload8(0, bn_scale + c_out_base) + vload8(0, bn_bias + c_out_base);
    float8 v1 = acc1 * vload8(1, bn_scale + c_out_base) + vload8(1, bn_bias + c_out_base);
    if (apply_relu) {
        v0 = fmax(v0, (float8)(0.0f));
        v1 = fmax(v1, (float8)(0.0f));
    }

    int base = voxel_idx * C_out + c_out_base;
    vstore8(convert_half8(v0), 0, output_features + base);
    vstore8(convert_half8(v1), 1, output_features + base);
}

// ============================================================
// Single-barrier local memory subm conv (FP16)
//
// One work-GROUP per voxel. WG_SIZE = C_out/8.
// Phase 1: Cooperatively load ALL 27 neighbor features into
//          local memory (one barrier)
// Phase 2: Compute all output channels from local memory
//          (no more global feature reads)
//
// gws = {N * WG, 1}, lws = {WG, 1} where WG = C_out/8
// ============================================================
__kernel void subm_conv_fp16_slm(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [27 * C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);

    if (voxel_idx >= N) return;

    // Phase 1: Cooperatively load ALL neighbor features into local memory
    int total_load = 27 * C_in;
    for (int t = lid; t < total_load; t += WG) {
        int k = t / C_in;
        int c = t % C_in;
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb >= 0) {
            local_feat[k * C_in + c] = input_features[nb * C_in + c];
        } else {
            local_feat[k * C_in + c] = (half)0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute from local memory
    int c_out_base = lid * 8;
    if (c_out_base >= C_out) return;

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    float s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    for (int k = 0; k < 27; k++) {
        int nb_k = neighbor_map[voxel_idx * 27 + k];
        if (nb_k < 0) continue;

        __local const half* f_row = local_feat + k * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(f_row[c]);
            int w_off = w_k_base + c * C_out;
            s0 += fv * convert_float(weights[w_off]);
            s1 += fv * convert_float(weights[w_off + 1]);
            s2 += fv * convert_float(weights[w_off + 2]);
            s3 += fv * convert_float(weights[w_off + 3]);
            s4 += fv * convert_float(weights[w_off + 4]);
            s5 += fv * convert_float(weights[w_off + 5]);
            s6 += fv * convert_float(weights[w_off + 6]);
            s7 += fv * convert_float(weights[w_off + 7]);
        }
    }

    // BN + ReLU
    float v0 = s0 * bn_scale[c_out_base]     + bn_bias[c_out_base];
    float v1 = s1 * bn_scale[c_out_base + 1] + bn_bias[c_out_base + 1];
    float v2 = s2 * bn_scale[c_out_base + 2] + bn_bias[c_out_base + 2];
    float v3 = s3 * bn_scale[c_out_base + 3] + bn_bias[c_out_base + 3];
    float v4 = s4 * bn_scale[c_out_base + 4] + bn_bias[c_out_base + 4];
    float v5 = s5 * bn_scale[c_out_base + 5] + bn_bias[c_out_base + 5];
    float v6 = s6 * bn_scale[c_out_base + 6] + bn_bias[c_out_base + 6];
    float v7 = s7 * bn_scale[c_out_base + 7] + bn_bias[c_out_base + 7];
    if (apply_relu) {
        v0 = fmax(v0, 0.0f); v1 = fmax(v1, 0.0f);
        v2 = fmax(v2, 0.0f); v3 = fmax(v3, 0.0f);
        v4 = fmax(v4, 0.0f); v5 = fmax(v5, 0.0f);
        v6 = fmax(v6, 0.0f); v7 = fmax(v7, 0.0f);
    }

    int base = voxel_idx * C_out + c_out_base;
    output_features[base]     = convert_half(v0);
    output_features[base + 1] = convert_half(v1);
    output_features[base + 2] = convert_half(v2);
    output_features[base + 3] = convert_half(v3);
    output_features[base + 4] = convert_half(v4);
    output_features[base + 5] = convert_half(v5);
    output_features[base + 6] = convert_half(v6);
    output_features[base + 7] = convert_half(v7);
}

// 
// Key optimization: each work-GROUP handles one voxel.
// - Phase 1: all work-items collaboratively load 27 neighbor feature
//   rows into local memory (shared across the group)
// - Phase 2: each work-item computes its output channel group using
//   fast local memory reads instead of repeated random global reads
//
// This reduces global memory reads by WG_SIZE× for features.
// ============================================================
// SIMD-friendly FP16 submanifold convolution
//
// Key: dim0 = output channel (varies fastest → adjacent SIMD lanes
// handle the SAME voxel, DIFFERENT channels).
// This gives:
//   - Shared input feature reads across SIMD lanes (L1 broadcast)
//   - Coalesced weight reads (adjacent c_out in memory)
//   - Each work-item handles 1 output channel
//
// Work sizing: gws = {C_out, N}
// ============================================================
__kernel void subm_conv_fp16_simd(
    __global const half* input_features,     // [N, C_in] - FP16
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out] - FP16
    __global const float* bn_scale,          // [C_out] - FP32
    __global const float* bn_bias,           // [C_out] - FP32
    __global half* output_features,          // [N, C_out] - FP16
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out = get_global_id(0);       // output channel (fastest dim → SIMD coalesced)
    int voxel_idx = get_global_id(1);   // voxel index

    if (voxel_idx >= N || c_out >= C_out) return;

    float acc = 0.0f;

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        // All SIMD lanes read the same neighbor features (L1 cache broadcast)
        __global const half* in_row = input_features + nb * C_in;
        // weights[k, c_in, c_out] — c_out varies fastest → coalesced across SIMD lanes
        int w_k_base = k * C_in * C_out + c_out;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            acc += fv * convert_float(weights[w_k_base + c * C_out]);
        }
    }

    // BN + optional ReLU
    float v = acc * bn_scale[c_out] + bn_bias[c_out];
    if (apply_relu) v = fmax(v, 0.0f);

    output_features[voxel_idx * C_out + c_out] = convert_half(v);
}


// Weights are still read from global but in a sequential pattern.
//
// Work-group layout:
//   dim0 = voxel index (one work-group per voxel)
//   dim1 = output channel group (each work-item handles 4 c_out)
//   work-group size in dim1 = C_out / 4
// ============================================================
__kernel void subm_conv_fp16_local(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_features             // [27 * C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(1);
    int wg_size = get_local_size(1);

    if (voxel_idx >= N) return;

    // --- Phase 1: Collaboratively load neighbor features into local memory ---
    // Load neighbor indices
    int nb_ids[27];
    int n_valid = 0;
    // Each work-item loads some neighbor IDs; with barrier this is shared
    for (int k = 0; k < 27; k++) {
        nb_ids[k] = neighbor_map[voxel_idx * 27 + k];
    }

    // Each work-item loads a portion of the 27*C_in values
    int total_load = 27 * C_in;
    for (int i = lid; i < total_load; i += wg_size) {
        int k = i / C_in;
        int c = i % C_in;
        int nb = nb_ids[k];
        if (nb >= 0) {
            local_features[i] = input_features[nb * C_in + c];
        } else {
            local_features[i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 2: Compute output channels ---
    int c_out_base = lid * 4;
    if (c_out_base >= C_out) return;

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;

    for (int k = 0; k < 27; k++) {
        if (nb_ids[k] < 0) continue;

        __local const half* feat_row = local_features + k * C_in;
        int w_k_base = k * C_in * C_out;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(feat_row[c]);
            int w_off = w_k_base + c * C_out + c_out_base;
            s0 += fv * convert_float(weights[w_off]);
            s1 += fv * convert_float(weights[w_off + 1]);
            s2 += fv * convert_float(weights[w_off + 2]);
            s3 += fv * convert_float(weights[w_off + 3]);
        }
    }

    // BN + optional ReLU
    float v0 = s0 * bn_scale[c_out_base]     + bn_bias[c_out_base];
    float v1 = s1 * bn_scale[c_out_base + 1] + bn_bias[c_out_base + 1];
    float v2 = s2 * bn_scale[c_out_base + 2] + bn_bias[c_out_base + 2];
    float v3 = s3 * bn_scale[c_out_base + 3] + bn_bias[c_out_base + 3];
    if (apply_relu) {
        v0 = fmax(v0, 0.0f); v1 = fmax(v1, 0.0f);
        v2 = fmax(v2, 0.0f); v3 = fmax(v3, 0.0f);
    }

    int base = voxel_idx * C_out + c_out_base;
    output_features[base]     = convert_half(v0);
    output_features[base + 1] = convert_half(v1);
    output_features[base + 2] = convert_half(v2);
    output_features[base + 3] = convert_half(v3);
}

// FP16 residual add + ReLU
__kernel void residual_add_relu_fp16(
    __global const half* a,
    __global const half* b,
    __global half* output,
    const int total_elements
) {
    int idx = get_global_id(0);
    if (idx >= total_elements) return;
    float val = convert_float(a[idx]) + convert_float(b[idx]);
    output[idx] = convert_half(val > 0.0f ? val : 0.0f);
}

// FP16 strided conv with scatter (atomic CAS on float output)
__kernel void strided_conv_scatter_fp16(
    __global const half* input_features,    // [N_in, C_in] - FP16
    __global const int* input_coords,       // [N_in, 4]
    __global const half* weights,           // [num_k, C_in, C_out] - FP16
    __global const float* bn_scale,         // [C_out] - FP32 (unused here, applied separately)
    __global const float* bn_bias,          // [C_out] - FP32
    __global float* output_features,        // [N_out, C_out] - FP32 (for atomic add)
    __global const int* out_hash_table,
    const int N_in, const int C_in, const int C_out,
    const int kx, const int ky, const int kz,
    const int sx, const int sy, const int sz,
    const int px, const int py, const int pz,
    const int X_out, const int Y_out, const int Z_out
) {
    int in_idx = get_global_id(0);
    int c_out = get_global_id(1);
    if (in_idx >= N_in || c_out >= C_out) return;

    int batch = input_coords[in_idx * 4 + 0];
    int in_x = input_coords[in_idx * 4 + 1];
    int in_y = input_coords[in_idx * 4 + 2];
    int in_z = input_coords[in_idx * 4 + 3];

    for (int ki = 0; ki < kx; ki++) {
        for (int kj = 0; kj < ky; kj++) {
            for (int kk = 0; kk < kz; kk++) {
                int ox = in_x + px - ki;
                int oy = in_y + py - kj;
                int oz = in_z + pz - kk;

                if (ox < 0 || ox % sx != 0 || oy < 0 || oy % sy != 0 || oz < 0 || oz % sz != 0)
                    continue;

                int out_x = ox / sx;
                int out_y = oy / sy;
                int out_z = oz / sz;

                if (out_x < 0 || out_x >= X_out || out_y < 0 || out_y >= Y_out || out_z < 0 || out_z >= Z_out)
                    continue;

                int hash_key = (out_x * Y_out + out_y) * Z_out + out_z;
                int out_idx = out_hash_table[hash_key];
                if (out_idx < 0) continue;

                int k_idx = (ki * ky + kj) * kz + kk;

                float sum = 0;
                for (int c = 0; c < C_in; c++) {
                    float fv = convert_float(input_features[in_idx * C_in + c]);
                    float wv = convert_float(weights[(k_idx * C_in + c) * C_out + c_out]);
                    sum += fv * wv;
                }

                // Atomic CAS float add
                __global volatile int* addr = (__global volatile int*)&output_features[out_idx * C_out + c_out];
                int expected, desired;
                do {
                    expected = *addr;
                    float old_val = as_float(expected);
                    float new_val = old_val + sum;
                    desired = as_int(new_val);
                } while (atomic_cmpxchg(addr, expected, desired) != expected);
            }
        }
    }
}

// FP16 strided conv using compact hash table (open-addressing)
// Replaces the dense hash table version to avoid 44MB upload per level
__kernel void strided_conv_scatter_fp16_compact(
    __global const half* input_features,    // [N_in, C_in] - FP16
    __global const int* input_coords,       // [N_in, 4]
    __global const half* weights,           // [num_k, C_in, C_out] - FP16
    __global const float* bn_scale,         // [C_out]
    __global const float* bn_bias,          // [C_out]
    __global float* output_features,        // [N_out, C_out] - FP32 (for atomic add)
    __global const int* hash_keys,          // compact hash table keys
    __global const int* hash_vals,          // compact hash table values
    const int N_in, const int C_in, const int C_out,
    const int kx, const int ky, const int kz,
    const int sx, const int sy, const int sz,
    const int px, const int py, const int pz,
    const int X_out, const int Y_out, const int Z_out,
    const int table_mask
) {
    int in_idx = get_global_id(0);
    int c_out = get_global_id(1);
    if (in_idx >= N_in || c_out >= C_out) return;

    int in_x = input_coords[in_idx * 4 + 1];
    int in_y = input_coords[in_idx * 4 + 2];
    int in_z = input_coords[in_idx * 4 + 3];

    for (int ki = 0; ki < kx; ki++) {
        for (int kj = 0; kj < ky; kj++) {
            for (int kk = 0; kk < kz; kk++) {
                int ox = in_x + px - ki;
                int oy = in_y + py - kj;
                int oz = in_z + pz - kk;

                if (ox < 0 || ox % sx != 0 || oy < 0 || oy % sy != 0 || oz < 0 || oz % sz != 0)
                    continue;

                int out_x = ox / sx;
                int out_y = oy / sy;
                int out_z = oz / sz;

                if (out_x < 0 || out_x >= X_out || out_y < 0 || out_y >= Y_out || out_z < 0 || out_z >= Z_out)
                    continue;

                // Compact hash table lookup (open addressing, linear probe)
                int encoded = (out_x * Y_out + out_y) * Z_out + out_z;
                int slot = (encoded * 2654435761u) & table_mask;
                int out_idx = -1;
                for (int p = 0; p <= table_mask; p++) {
                    int k = hash_keys[slot];
                    if (k == encoded) { out_idx = hash_vals[slot]; break; }
                    if (k == -1) break;
                    slot = (slot + 1) & table_mask;
                }
                if (out_idx < 0) continue;

                int k_idx = (ki * ky + kj) * kz + kk;

                float sum = 0;
                for (int c = 0; c < C_in; c++) {
                    float fv = convert_float(input_features[in_idx * C_in + c]);
                    float wv = convert_float(weights[(k_idx * C_in + c) * C_out + c_out]);
                    sum += fv * wv;
                }

                // Atomic CAS float add
                __global volatile int* addr = (__global volatile int*)&output_features[out_idx * C_out + c_out];
                int expected, desired;
                do {
                    expected = *addr;
                    float old_val = as_float(expected);
                    float new_val = old_val + sum;
                    desired = as_int(new_val);
                } while (atomic_cmpxchg(addr, expected, desired) != expected);
            }
        }
    }
}

// ============================================================
// Gather-based strided conv: each output computes its value
// by reading from input positions via INPUT hash table.
// No atomics needed! BN+ReLU fused, outputs FP16 directly.
// gws = {N_out, C_out}
// ============================================================
__kernel void strided_conv_gather_fp16(
    __global const half* input_features,    // [N_in, C_in] - FP16
    __global const int* output_coords,      // [N_out, 4]
    __global const half* weights,           // [num_k, C_in, C_out] - FP16
    __global const float* bn_scale,         // [C_out]
    __global const float* bn_bias,          // [C_out]
    __global half* output_features,         // [N_out, C_out] - FP16 direct!
    __global const int* hash_keys,          // INPUT hash table keys
    __global const int* hash_vals,          // INPUT hash table values
    const int N_out, const int C_in, const int C_out,
    const int kx, const int ky, const int kz,
    const int sx, const int sy, const int sz,
    const int px, const int py, const int pz,
    const int X_in, const int Y_in, const int Z_in,
    const int table_mask
) {
    int out_idx = get_global_id(0);
    int c_out = get_global_id(1);
    if (out_idx >= N_out || c_out >= C_out) return;

    int out_x = output_coords[out_idx * 4 + 1];
    int out_y = output_coords[out_idx * 4 + 2];
    int out_z = output_coords[out_idx * 4 + 3];

    float sum = 0.0f;

    for (int ki = 0; ki < kx; ki++) {
        int in_x = out_x * sx - px + ki;
        if (in_x < 0 || in_x >= X_in) continue;

        for (int kj = 0; kj < ky; kj++) {
            int in_y = out_y * sy - py + kj;
            if (in_y < 0 || in_y >= Y_in) continue;

            for (int kk = 0; kk < kz; kk++) {
                int in_z = out_z * sz - pz + kk;
                if (in_z < 0 || in_z >= Z_in) continue;

                // Look up input position in INPUT hash table
                int encoded = (in_x * Y_in + in_y) * Z_in + in_z;
                int slot = (encoded * 2654435761u) & table_mask;
                int in_idx = -1;
                for (int p = 0; p <= table_mask; p++) {
                    int k = hash_keys[slot];
                    if (k == encoded) { in_idx = hash_vals[slot]; break; }
                    if (k == -1) break;
                    slot = (slot + 1) & table_mask;
                }
                if (in_idx < 0) continue;

                int k_idx = (ki * ky + kj) * kz + kk;

                for (int c = 0; c < C_in; c++) {
                    float fv = convert_float(input_features[in_idx * C_in + c]);
                    float wv = convert_float(weights[(k_idx * C_in + c) * C_out + c_out]);
                    sum += fv * wv;
                }
            }
        }
    }

    // Fused BN + ReLU + FP16 output
    sum = sum * bn_scale[c_out] + bn_bias[c_out];
    if (sum < 0.0f) sum = 0.0f;
    output_features[out_idx * C_out + c_out] = convert_half(sum);
}

// ============================================================
// Build strided conv neighbor map on GPU
// For each output voxel, lookup which input voxels contribute
// using the INPUT hash table. Stores result in nmap[N_out, num_k].
// This precomputation eliminates hash lookups from the conv kernel.
// ============================================================
__kernel void build_strided_nmap(
    __global const int* output_coords,      // [N_out, 4]
    __global const int* hash_keys,          // INPUT hash table keys
    __global const int* hash_vals,          // INPUT hash table values
    __global int* nmap,                     // [N_out, num_k] output
    const int N_out,
    const int kx, const int ky, const int kz,
    const int sx, const int sy, const int sz,
    const int px, const int py, const int pz,
    const int X_in, const int Y_in, const int Z_in,
    const int table_mask
) {
    int out_idx = get_global_id(0);
    if (out_idx >= N_out) return;

    int out_x = output_coords[out_idx * 4 + 1];
    int out_y = output_coords[out_idx * 4 + 2];
    int out_z = output_coords[out_idx * 4 + 3];

    int k = 0;
    for (int ki = 0; ki < kx; ki++) {
        int in_x = out_x * sx - px + ki;
        for (int kj = 0; kj < ky; kj++) {
            int in_y = out_y * sy - py + kj;
            for (int kk = 0; kk < kz; kk++) {
                int in_z = out_z * sz - pz + kk;
                int result = -1;

                if (in_x >= 0 && in_x < X_in &&
                    in_y >= 0 && in_y < Y_in &&
                    in_z >= 0 && in_z < Z_in) {
                    int encoded = (in_x * Y_in + in_y) * Z_in + in_z;
                    uint h = ((uint)encoded * 2654435761u) & (uint)table_mask;
                    while (true) {
                        int stored = hash_keys[h];
                        if (stored == encoded) { result = hash_vals[h]; break; }
                        if (stored == -1) break;
                        h = (h + 1) & (uint)table_mask;
                    }
                }
                nmap[out_idx * (kx * ky * kz) + k] = result;
                k++;
            }
        }
    }
}

// ============================================================
// Strided conv using precomputed neighbor map (no hash lookups!)
// w4 vectorized: 4 output channels per WI.
// gws = {C_out/4, N_out}
// ============================================================
__kernel void strided_conv_nmap_fp16(
    __global const half* input_features,    // [N_in, C_in]
    __global const int* nmap,               // [N_out, num_k]
    __global const half* weights,           // [num_k, C_in, C_out]
    __global const float* bn_scale,         // [C_out]
    __global const float* bn_bias,          // [C_out]
    __global half* output_features,         // [N_out, C_out]
    const int N_out,
    const int C_in,
    const int C_out,
    const int num_k,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 4;
    int out_idx = get_global_id(1);
    if (out_idx >= N_out || c_out_base >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < num_k; k++) {
        int in_idx = nmap[out_idx * num_k + k];
        if (in_idx < 0) continue;

        __global const half* in_row = input_features + in_idx * C_in;
        int wb = k * C_in * C_out + c_out_base;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            acc += fv * convert_float4(vload4(0, weights + wb + c * C_out));
        }
    }

    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float4)(0.0f));

    vstore4(convert_half4(v), 0, output_features + out_idx * C_out + c_out_base);
}

// ============================================================
// SLM v2 subm conv: vectorized weight reads + SLM features
//
// One work-GROUP per voxel. Phase 1: cooperatively load all
// neighbor features into SLM. Phase 2: compute output channels
// using SLM features + global weights (vload4 vectorized).
// 4 output channels per WI.
// gws = {N * C_out/4}, lws = {C_out/4}
// ============================================================
__kernel void subm_conv_fp16_slm_v2(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [27 * C_in]
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);

    if (voxel_idx >= N) return;

    // Preload neighbor indices into registers
    int nb[27];
    for (int k = 0; k < 27; k++) {
        nb[k] = neighbor_map[voxel_idx * 27 + k];
    }

    // Phase 1: Cooperatively load ALL neighbor features into SLM
    int total_load = 27 * C_in;
    for (int i = lid; i < total_load; i += WG) {
        int k = i / C_in;
        int c = i % C_in;
        local_feat[i] = (nb[k] >= 0) ? input_features[nb[k] * C_in + c] : (half)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute 4 output channels from SLM + global weights
    int co = lid * 4;
    if (co >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;

        __local const half* f_row = local_feat + k * C_in;
        int wb = k * C_in * C_out + co;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(f_row[c]);
            acc += fv * convert_float4(vload4(0, weights + wb + c * C_out));
        }
    }

    // Fused BN + ReLU
    float4 sc = vload4(0, bn_scale + co);
    float4 bi = vload4(0, bn_bias + co);
    float4 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float4)(0.0f));

    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + co);
}

// ============================================================
// SLM v2 w8: 8 output channels per WI + SLM feature sharing
// gws = {N * C_out/8}, lws = {C_out/8}
// ============================================================
__kernel void subm_conv_fp16_slm_v2_w8(
    __global const half* input_features,
    __global const int* neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat
) {
    int voxel_idx = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);

    if (voxel_idx >= N) return;

    int nb[27];
    for (int k = 0; k < 27; k++) {
        nb[k] = neighbor_map[voxel_idx * 27 + k];
    }

    int total_load = 27 * C_in;
    for (int i = lid; i < total_load; i += WG) {
        int k = i / C_in;
        int c = i % C_in;
        local_feat[i] = (nb[k] >= 0) ? input_features[nb[k] * C_in + c] : (half)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int co = lid * 8;
    if (co >= C_out) return;

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;

        __local const half* f_row = local_feat + k * C_in;
        int wb = k * C_in * C_out + co;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(f_row[c]);
            acc += fv * convert_float8(vload8(0, weights + wb + c * C_out));
        }
    }

    float8 sc = vload8(0, bn_scale + co);
    float8 bi = vload8(0, bn_bias + co);
    float8 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float8)(0.0f));

    vstore8(convert_half8(v), 0, output_features + voxel_idx * C_out + co);
}

// ============================================================
// Multi-voxel SLM: process NV voxels per work-group for
// better occupancy. 4 output channels per WI.
// WG = NV * (C_out/4). gws = {ceil(N/NV) * WG}, lws = {WG}
// NV=4 for C_in<=128: SLM = 4 * 27 * 128 * 2 = 27648B (fits 64KB)
// ============================================================
#define MV_NV 4
__kernel void subm_conv_fp16_mv4(
    __global const half* input_features,
    __global const int* neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu,
    __local half* local_feat                 // [MV_NV * 27 * C_in]
) {
    int group_id = get_group_id(0);
    int lid = get_local_id(0);
    int WG = get_local_size(0);         // = MV_NV * (C_out / 4)
    int voxels_per_wi = C_out / 4;      // WIs per voxel

    // Which voxel within the group does this WI process?
    int local_vi = lid / voxels_per_wi;
    int local_co_idx = lid % voxels_per_wi;
    int voxel_idx = group_id * MV_NV + local_vi;

    // Load neighbor indices for this WI's voxel
    int nb[27];
    if (voxel_idx < N) {
        for (int k = 0; k < 27; k++)
            nb[k] = neighbor_map[voxel_idx * 27 + k];
    } else {
        for (int k = 0; k < 27; k++) nb[k] = -1;
    }

    // Phase 1: Cooperatively load ALL MV_NV voxels' features into SLM
    int total_load = MV_NV * 27 * C_in;
    int feat_per_voxel = 27 * C_in;
    for (int i = lid; i < total_load; i += WG) {
        int vi = i / feat_per_voxel;
        int rem = i % feat_per_voxel;
        int k = rem / C_in;
        int c = rem % C_in;
        int glob_vi = group_id * MV_NV + vi;
        int n_idx = -1;
        if (glob_vi < N) {
            n_idx = neighbor_map[glob_vi * 27 + k];
        }
        local_feat[i] = (n_idx >= 0) ? input_features[n_idx * C_in + c] : (half)0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (voxel_idx >= N) return;

    // Phase 2: Compute 4 output channels
    int co = local_co_idx * 4;
    if (co >= C_out) return;

    float4 acc = (float4)(0.0f);
    __local const half* my_feat = local_feat + local_vi * feat_per_voxel;

    for (int k = 0; k < 27; k++) {
        if (nb[k] < 0) continue;
        __local const half* f_row = my_feat + k * C_in;
        int wb = k * C_in * C_out + co;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(f_row[c]);
            acc += fv * convert_float4(vload4(0, weights + wb + c * C_out));
        }
    }

    float4 sc = vload4(0, bn_scale + co);
    float4 bi = vload4(0, bn_bias + co);
    float4 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float4)(0.0f));

    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + co);
}

// Sparse to dense with FP16 input
__kernel void sparse_to_dense_fp16(
    __global const half* features,
    __global const int* coords,
    __global float* dense,
    const int N, const int C, const int X, const int Y, const int Z
) {
    int idx = get_global_id(0);
    int c = get_global_id(1);
    if (idx >= N || c >= C) return;

    int batch = coords[idx * 4 + 0];
    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];

    if (x >= 0 && x < X && y >= 0 && y < Y && z >= 0 && z < Z) {
        int dense_idx = ((((batch * C + c) * X + x) * Y + y) * Z + z);
        dense[dense_idx] = convert_float(features[idx * C + c]);
    }
}

// ============================================================
// OPTIMIZED SUBM CONV KERNELS
// ============================================================

// ── Kernel A: XMX/DPAS-based subm conv ──
// Uses intel_sub_group_f16_f16_matrix_mad_k16 for hardware matrix multiply
// Each sub-group of 16 WIs handles 16 voxels × 8 output channels
// DPAS computes [16,16] × [16,8] → [16,8] per call
// Weights must be in VNNI-packed format: [27, C_in/16, C_out/8, 8, 16] as uint
// gws = {C_out/8, ceil(N/16)*16}, lws = {1, 16}
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroup_matrix_multiply_accumulate

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subm_conv_fp16_dpas(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const uint* weights_packed,     // VNNI format: [27, C_in/16, C_out/8, 8, 16] as uint
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_tile = get_global_id(0);   // which 8-output-channel tile
    int voxel_base = get_group_id(1) * 16;
    int sg_lid = get_sub_group_local_id();  // 0..15 within sub-group
    
    int my_voxel = voxel_base + sg_lid;
    int c_out_base = c_out_tile * 8;
    
    if (c_out_base >= C_out) return;
    
    int C_in_tiles = C_in / 16;
    int C_out_tiles = C_out / 8;
    
    float8 acc = (float8)(0.0f);
    
    for (int k = 0; k < 27; k++) {
        int nb = (my_voxel < N) ? neighbor_map[my_voxel * 27 + k] : -1;
        
        // Skip if no WI has a valid neighbor (use sub-group reduce)
        int any_valid = sub_group_reduce_max(nb >= 0 ? 1 : 0);
        if (any_valid == 0) continue;
        
        for (int ci_tile = 0; ci_tile < C_in_tiles; ci_tile++) {
            // Load A: each WI loads 16 half features from its neighbor
            // Pack as int8 (8 uints = 16 halfs)
            int8 a;
            if (nb >= 0) {
                __global const uint* feat_ptr = (__global const uint*)(input_features + nb * C_in + ci_tile * 16);
                a = (int8)(feat_ptr[0], feat_ptr[1], feat_ptr[2], feat_ptr[3],
                          feat_ptr[4], feat_ptr[5], feat_ptr[6], feat_ptr[7]);
            } else {
                a = (int8)(0);
            }
            
            // Load B: weight tile in VNNI format via sub-group block read
            int w_offset = ((k * C_in_tiles + ci_tile) * C_out_tiles + c_out_tile) * 128;
            int8 b = as_int8(intel_sub_group_block_read8(weights_packed + w_offset));
            
            // XMX: acc += A * B
            acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
        }
    }
    
    // Apply BN + ReLU (each WI writes 8 output channels for its voxel)
    if (my_voxel < N) {
        __global const float* sc_ptr = bn_scale + c_out_base;
        __global const float* bi_ptr = bn_bias + c_out_base;
        float8 sc = (float8)(sc_ptr[0], sc_ptr[1], sc_ptr[2], sc_ptr[3],
                            sc_ptr[4], sc_ptr[5], sc_ptr[6], sc_ptr[7]);
        float8 bi = (float8)(bi_ptr[0], bi_ptr[1], bi_ptr[2], bi_ptr[3],
                            bi_ptr[4], bi_ptr[5], bi_ptr[6], bi_ptr[7]);
        float8 v = acc * sc + bi;
        if (apply_relu) v = fmax(v, (float8)(0.0f));
        vstore8(convert_half8(v), 0, output_features + my_voxel * C_out + c_out_base);
    }
}

// ── Kernel B: DPAS with residual fuse ──
// Same as dpas but adds residual (identity) and applies ReLU
// For the second conv in a residual block + add + relu, fused
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subm_conv_fp16_dpas_resadd(
    __global const half* input_features,     // [N, C]
    __global const int* neighbor_map,        // [N, 27]
    __global const uint* weights_packed,     // VNNI format
    __global const float* bn_scale,          // [C]
    __global const float* bn_bias,           // [C]
    __global half* output_features,          // [N, C] - output (also identity source)
    __global const half* identity_features,  // [N, C] - first conv input (identity)
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_tile = get_global_id(0);
    int voxel_base = get_group_id(1) * 16;
    int sg_lid = get_sub_group_local_id();
    
    int my_voxel = voxel_base + sg_lid;
    int c_out_base = c_out_tile * 8;
    
    if (c_out_base >= C_out) return;
    
    int C_in_tiles = C_in / 16;
    int C_out_tiles = C_out / 8;
    
    float8 acc = (float8)(0.0f);
    
    for (int k = 0; k < 27; k++) {
        int nb = (my_voxel < N) ? neighbor_map[my_voxel * 27 + k] : -1;
        int any_valid = sub_group_reduce_max(nb >= 0 ? 1 : 0);
        if (any_valid == 0) continue;
        
        for (int ci_tile = 0; ci_tile < C_in_tiles; ci_tile++) {
            int8 a;
            if (nb >= 0) {
                __global const uint* feat_ptr = (__global const uint*)(input_features + nb * C_in + ci_tile * 16);
                a = (int8)(feat_ptr[0], feat_ptr[1], feat_ptr[2], feat_ptr[3],
                          feat_ptr[4], feat_ptr[5], feat_ptr[6], feat_ptr[7]);
            } else {
                a = (int8)(0);
            }
            
            int w_offset = ((k * C_in_tiles + ci_tile) * C_out_tiles + c_out_tile) * 128;
            int8 b = as_int8(intel_sub_group_block_read8(weights_packed + w_offset));
            acc = intel_sub_group_f16_f16_matrix_mad_k16(a, b, acc);
        }
    }
    
    // BN + residual add + ReLU
    if (my_voxel < N) {
        __global const float* sc_ptr = bn_scale + c_out_base;
        __global const float* bi_ptr = bn_bias + c_out_base;
        float8 sc = (float8)(sc_ptr[0], sc_ptr[1], sc_ptr[2], sc_ptr[3],
                            sc_ptr[4], sc_ptr[5], sc_ptr[6], sc_ptr[7]);
        float8 bi = (float8)(bi_ptr[0], bi_ptr[1], bi_ptr[2], bi_ptr[3],
                            bi_ptr[4], bi_ptr[5], bi_ptr[6], bi_ptr[7]);
        float8 v = acc * sc + bi;
        
        // Add identity (residual connection)
        half8 id_h = vload8(0, identity_features + my_voxel * C_out + c_out_base);
        v += convert_float8(id_h);
        
        if (apply_relu) v = fmax(v, (float8)(0.0f));
        vstore8(convert_half8(v), 0, output_features + my_voxel * C_out + c_out_base);
    }
}
#endif  // cl_intel_subgroup_matrix_multiply_accumulate


// ── Kernel C: Optimized w4 with FP16 accumulation + prefetch ──
// Uses native FP16 compute (2× throughput) with FP32 BN at the end
// Also preloads neighbor indices and uses sub_group_broadcast for features
// gws = {C_out/4, N}

// ============================================================
// "Lite" subm conv: FP16 accumulation without neighbor preloading.
// Combines w4's low register pressure with FP16 compute throughput.
// The inner C_in loop stays in half precision, flushing to FP32
// per-neighbor to maintain accuracy.
// gws = {C_out/4, N}
// ============================================================
__kernel void subm_conv_fp16_lite(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 4;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        half4 hacc = (half4)(0.0h);
        for (int c = 0; c < C_in; c++) {
            half fv = in_row[c];
            half4 w4 = vload4(0, weights + w_k_base + c * C_out);
            hacc += fv * w4;
        }
        acc += convert_float4(hacc);
    }

    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float4)(0.0f));
    }
    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + c_out_base);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subm_conv_fp16_fast(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 4;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    // Preload all 27 neighbor indices into private memory
    int nb[27];
    int n_valid = 0;
    int valid_k[27];
    int valid_nb[27];
    for (int k = 0; k < 27; k++) {
        nb[k] = neighbor_map[voxel_idx * 27 + k];
        if (nb[k] >= 0) {
            valid_k[n_valid] = k;
            valid_nb[n_valid] = nb[k];
            n_valid++;
        }
    }

    // FP16 accumulation with periodic FP32 flush to prevent overflow
    float4 acc = (float4)(0.0f);
    
    for (int vi = 0; vi < n_valid; vi++) {
        int k = valid_k[vi];
        int n = valid_nb[vi];
        
        __global const half* in_row = input_features + n * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        half4 hacc = (half4)(0.0h);
        for (int c = 0; c < C_in; c++) {
            half fv = in_row[c];
            half4 w4 = vload4(0, weights + w_k_base + c * C_out);
            hacc += fv * w4;
        }
        acc += convert_float4(hacc);  // flush to FP32 per-neighbor
    }

    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float4)(0.0f));
    }
    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + c_out_base);
}


// ── Kernel D: Morton-ordered processing with sub-group features sharing ──
// Processes voxels in Morton (Z-order) curve order for better cache locality
// Adjacent work items in dim1 process spatially nearby voxels
// gws = {C_out/4, N}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void subm_conv_fp16_morton(
    __global const half* input_features,
    __global const int* neighbor_map,
    __global const half* weights,
    __global const float* bn_scale,
    __global const float* bn_bias,
    __global half* output_features,
    __global const int* morton_order,    // [N] permutation: work_idx -> voxel_idx
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 4;
    int work_idx = get_global_id(1);

    if (work_idx >= N || c_out_base >= C_out) return;
    
    int voxel_idx = morton_order[work_idx];

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        for (int c = 0; c < C_in; c++) {
            float fv = convert_float(in_row[c]);
            half4 w4 = vload4(0, weights + w_k_base + c * C_out);
            acc += fv * convert_float4(w4);
        }
    }

    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 v = acc * sc + bi;
    if (apply_relu) v = fmax(v, (float4)(0.0f));
    vstore4(convert_half4(v), 0, output_features + voxel_idx * C_out + c_out_base);
}


// ── Kernel E: Feature permutation for Morton reordering ──
// Reorders features according to a permutation: out[i] = in[perm[i]]
// gws = {N * C / 8}  (vectorized half8 copy)
__kernel void permute_features_fp16(
    __global const half* in_features,   // [N, C]
    __global const int* perm,           // [N] permutation
    __global half* out_features,        // [N, C]
    const int N,
    const int C
) {
    int flat = get_global_id(0);
    int voxel = flat / (C / 8);
    int c_group = flat % (C / 8);
    if (voxel >= N) return;
    
    int src = perm[voxel];
    half8 val = vload8(c_group, in_features + src * C);
    vstore8(val, c_group, out_features + voxel * C);
}

// ============================================================
// Fused sparse-to-BEV kernel with FP16 input
//
// Directly writes sparse features to BEV layout:
//   [1, C*Z, X, Y] where bev_channel = c*Z + z
//
// This eliminates:
//   1. Intermediate dense [C,X,Y,Z] buffer
//   2. GPU readback of the full dense tensor
//   3. CPU permutation loop (millions of iterations)
// ============================================================
__kernel void sparse_to_bev_fp16(
    __global const half* features,    // [N, C]
    __global const int* coords,       // [N, 4]: batch, x, y, z
    __global float* bev_output,       // [1, C*Z, X, Y] - pre-zeroed
    const int N, const int C,
    const int X, const int Y, const int Z
) {
    int idx = get_global_id(0);
    int c = get_global_id(1);
    if (idx >= N || c >= C) return;

    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];

    if (x >= 0 && x < X && y >= 0 && y < Y && z >= 0 && z < Z) {
        int bev_c = c * Z + z;
        int bev_idx = (bev_c * X + x) * Y + y;
        bev_output[bev_idx] = convert_float(features[idx * C + c]);
    }
}

// ============================================================
// Sparse→BEV scatter: writes FP16 directly (no half→float conversion).
// Output buffer is half the size of FP32 variant (16.5MB vs 33MB),
// saving ~50% readback bandwidth on iGPU.
// CPU does the FP16→FP32 expansion after readback.
// gws = {N, C}
// ============================================================
__kernel void sparse_to_bev_fp16_out(
    __global const half* features,    // [N, C]
    __global const int* coords,       // [N, 4]: batch, x, y, z
    __global half* bev_output,        // [1, C*Z, X, Y] FP16 - pre-zeroed
    const int N, const int C,
    const int X, const int Y, const int Z
) {
    int idx = get_global_id(0);
    int c = get_global_id(1);
    if (idx >= N || c >= C) return;

    int x = coords[idx * 4 + 1];
    int y = coords[idx * 4 + 2];
    int z = coords[idx * 4 + 3];

    if (x >= 0 && x < X && y >= 0 && y < Y && z >= 0 && z < Z) {
        int bev_c = c * Z + z;
        int bev_idx = (bev_c * X + x) * Y + y;
        bev_output[bev_idx] = features[idx * C + c];
    }
}

// ============================================================
// Optimized lite kernel processing 8 output channels per WI
// Uses FP16 accumulation for speed, FP32 only for BN.
// Halves work-items vs lite (4-wide), reducing nmap reads by 2×.
// gws = {C_out/8, N}
// ============================================================
__kernel void subm_conv_fp16_lite8(
    __global const half* input_features,     // [N, C_in]
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out]
    const int N,
    const int C_in,
    const int C_out,
    const int apply_relu
) {
    int c_out_base = get_global_id(0) * 8;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    // Load neighbor indices once into private memory
    int nb_idx[27];
    for (int k = 0; k < 27; k++)
        nb_idx[k] = neighbor_map[voxel_idx * 27 + k];

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = nb_idx[k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        __global const half* w_row = weights + k * C_in * C_out + c_out_base;

        half8 hacc = (half8)(0.0h);
        int c = 0;
        // Unrolled loop: process 4 input channels per iteration
        for (; c + 3 < C_in; c += 4) {
            half fv0 = in_row[c];
            half fv1 = in_row[c+1];
            half fv2 = in_row[c+2];
            half fv3 = in_row[c+3];
            hacc += fv0 * vload8(0, w_row + c * C_out);
            hacc += fv1 * vload8(0, w_row + (c+1) * C_out);
            hacc += fv2 * vload8(0, w_row + (c+2) * C_out);
            hacc += fv3 * vload8(0, w_row + (c+3) * C_out);
        }
        for (; c < C_in; c++) {
            hacc += in_row[c] * vload8(0, w_row + c * C_out);
        }
        acc += convert_float8(hacc);
    }

    float8 sc = vload8(0, bn_scale + c_out_base);
    float8 bi = vload8(0, bn_bias + c_out_base);
    float8 v = acc * sc + bi;
    if (apply_relu) {
        v = fmax(v, (float8)(0.0f));
    }
    vstore8(convert_half8(v), 0, output_features + voxel_idx * C_out + c_out_base);
}

// ============================================================
// Fused conv2 + residual add + ReLU kernel
// Eliminates extra global memory write/read for residual path
// gws = {C_out/4, N}
// ============================================================
__kernel void subm_conv_fp16_lite_resadd(
    __global const half* input_features,     // [N, C_in] - conv2 input
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out] - final residual output
    __global const half* identity,           // [N, C_out] - identity (residual input)
    const int N,
    const int C_in,
    const int C_out
) {
    int c_out_base = get_global_id(0) * 4;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    float4 acc = (float4)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = neighbor_map[voxel_idx * 27 + k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        int w_k_base = k * C_in * C_out + c_out_base;

        half4 hacc = (half4)(0.0h);
        for (int c = 0; c < C_in; c++) {
            half fv = in_row[c];
            half4 w4 = vload4(0, weights + w_k_base + c * C_out);
            hacc += fv * w4;
        }
        acc += convert_float4(hacc);
    }

    // BN (no ReLU) + residual add + ReLU
    float4 sc = vload4(0, bn_scale + c_out_base);
    float4 bi = vload4(0, bn_bias + c_out_base);
    float4 conv_out = acc * sc + bi;  // BN without ReLU
    
    // Add identity (residual connection) then ReLU
    int offset = voxel_idx * C_out + c_out_base;
    float4 id = convert_float4(vload4(0, identity + offset));
    float4 v = fmax(conv_out + id, (float4)(0.0f));
    vstore4(convert_half4(v), 0, output_features + offset);
}

// ============================================================
// Fused conv2 + residual add + ReLU kernel (8-wide)
// gws = {C_out/8, N}
// ============================================================
__kernel void subm_conv_fp16_lite8_resadd(
    __global const half* input_features,     // [N, C_in] - conv2 input
    __global const int* neighbor_map,        // [N, 27]
    __global const half* weights,            // [27, C_in, C_out]
    __global const float* bn_scale,          // [C_out]
    __global const float* bn_bias,           // [C_out]
    __global half* output_features,          // [N, C_out] - final residual output
    __global const half* identity,           // [N, C_out] - identity (residual input)
    const int N,
    const int C_in,
    const int C_out
) {
    int c_out_base = get_global_id(0) * 8;
    int voxel_idx = get_global_id(1);

    if (voxel_idx >= N || c_out_base >= C_out) return;

    // Load neighbor indices once
    int nb_idx[27];
    for (int k = 0; k < 27; k++)
        nb_idx[k] = neighbor_map[voxel_idx * 27 + k];

    float8 acc = (float8)(0.0f);

    for (int k = 0; k < 27; k++) {
        int nb = nb_idx[k];
        if (nb < 0) continue;

        __global const half* in_row = input_features + nb * C_in;
        __global const half* w_row = weights + k * C_in * C_out + c_out_base;

        half8 hacc = (half8)(0.0h);
        int c = 0;
        for (; c + 3 < C_in; c += 4) {
            half fv0 = in_row[c];
            half fv1 = in_row[c+1];
            half fv2 = in_row[c+2];
            half fv3 = in_row[c+3];
            hacc += fv0 * vload8(0, w_row + c * C_out);
            hacc += fv1 * vload8(0, w_row + (c+1) * C_out);
            hacc += fv2 * vload8(0, w_row + (c+2) * C_out);
            hacc += fv3 * vload8(0, w_row + (c+3) * C_out);
        }
        for (; c < C_in; c++) {
            hacc += in_row[c] * vload8(0, w_row + c * C_out);
        }
        acc += convert_float8(hacc);
    }

    // BN (no ReLU) + residual add + ReLU
    float8 sc = vload8(0, bn_scale + c_out_base);
    float8 bi = vload8(0, bn_bias + c_out_base);
    float8 conv_out = acc * sc + bi;

    int offset = voxel_idx * C_out + c_out_base;
    float8 id = convert_float8(vload8(0, identity + offset));
    float8 v = fmax(conv_out + id, (float8)(0.0f));
    vstore8(convert_half8(v), 0, output_features + offset);
}

// ============================================================
// GPU Keepalive Kernel
// ============================================================
// Lightweight kernel that keeps the GPU active to prevent frequency
// down-scaling between pipeline stages. Uses a single work-item
// with a calibrated busy-loop so EU usage is minimal (<1% of 96 EUs).
__kernel void gpu_keepalive(__global int* buf, int iters) {
    int x = buf[0];
    for (int i = 0; i < iters; i++) {
        x = (x * 1103515245 + 12345);
    }
    buf[0] = x;
}
