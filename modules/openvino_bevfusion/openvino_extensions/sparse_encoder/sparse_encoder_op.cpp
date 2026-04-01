// SparseEncoder OpenVINO Extension — Full Implementation
//
// Ports the fused GPU sparse encoder from sparse_conv_fused.cpp
// to an OpenVINO custom op without torch dependency.
// Uses OpenCL for GPU kernels, CPU for neighbor map construction.

#include "sparse_encoder_op.hpp"
#include <openvino/core/type.hpp>
#include <openvino/core/shape.hpp>

#include <CL/cl.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <chrono>
#include <mutex>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace BEVFusionExtension {

// ============================================================
// OpenCL State (lazy-initialized, thread-safe)
// ============================================================
static std::once_flag g_init_flag;
static cl_context     g_context  = nullptr;
static cl_command_queue g_queue  = nullptr;
static cl_device_id   g_device   = nullptr;
static cl_program     g_program  = nullptr;

// FP16 kernels
static cl_kernel g_subm_conv_fp16_kernel         = nullptr;
static cl_kernel g_subm_conv_fp16_scalar_kernel  = nullptr;
static cl_kernel g_subm_conv_fp16_w4_kernel      = nullptr;
static cl_kernel g_subm_conv_fp16_slm_kernel     = nullptr;
static cl_kernel g_subm_conv_fp16_w16_kernel     = nullptr;
static cl_kernel g_subm_conv_fp16_simd_kernel    = nullptr;
static cl_kernel g_subm_conv_fp16_local_kernel   = nullptr;
static cl_kernel g_residual_add_relu_fp16_kernel  = nullptr;
static cl_kernel g_strided_conv_fp16_kernel       = nullptr;
static cl_kernel g_strided_conv_fp16_compact_kernel = nullptr;
static cl_kernel g_strided_conv_gather_fp16_kernel = nullptr;
static cl_kernel g_apply_bn_relu_kernel           = nullptr;
static cl_kernel g_float_to_half_kernel           = nullptr;
static cl_kernel g_sparse_to_dense_fp16_kernel    = nullptr;

// Persistent GPU buffers
static cl_mem g_feat_buf_a       = nullptr;
static cl_mem g_feat_buf_b       = nullptr;
static cl_mem g_identity_buf     = nullptr;
static cl_mem g_neighbor_map_buf = nullptr;
static cl_mem g_coords_buf       = nullptr;
static cl_mem g_weights_buf      = nullptr;
static cl_mem g_scale_buf        = nullptr;
static cl_mem g_bias_buf         = nullptr;
static cl_mem g_temp_fp32_buf    = nullptr;
static cl_mem g_bev_buf          = nullptr;
static cl_mem g_bev_fp16_buf     = nullptr;  // FP16 BEV for reduced readback
static cl_mem g_strided_hash_buf = nullptr;
static cl_mem g_strided_feat_buf = nullptr;
static size_t g_fab_sz = 0, g_fbb_sz = 0, g_idb_sz = 0;
static size_t g_nmb_sz = 0, g_cdb_sz = 0;
static size_t g_wb_sz = 0, g_sb_sz = 0, g_bb_sz = 0, g_t32_sz = 0;
static size_t g_bev_sz = 0, g_bev_fp16_sz = 0, g_shb_sz = 0, g_sfb_sz = 0;

// Pre-uploaded per-layer weight buffers (optimization: convert FP16 once)
constexpr int MAX_LAYERS = 21;
static cl_mem g_layer_weights[MAX_LAYERS] = {};
static cl_mem g_layer_scales[MAX_LAYERS]  = {};
static cl_mem g_layer_biases[MAX_LAYERS]  = {};
static bool   g_weights_uploaded = false;

// SLM v2 kernels (vectorized weight reads + shared features)
static cl_kernel g_subm_conv_fp16_slm_v2_kernel    = nullptr;
static cl_kernel g_subm_conv_fp16_slm_v2_w8_kernel = nullptr;
static cl_kernel g_subm_conv_fp16_mv4_kernel        = nullptr;

// Strided conv with precomputed neighbor map (no hash lookups)
static cl_kernel g_build_strided_nmap_kernel = nullptr;
static cl_kernel g_strided_conv_nmap_fp16_kernel = nullptr;

// NEW: DPAS/XMX-based kernels (hardware matrix multiply)
static cl_kernel g_subm_conv_fp16_dpas_kernel = nullptr;
static cl_kernel g_subm_conv_fp16_dpas_resadd_kernel = nullptr;
// NEW: Optimized FP16 accumulation + prefetch kernel
static cl_kernel g_subm_conv_fp16_fast_kernel = nullptr;
// NEW: FP16 accumulation without preloading (low register pressure)
static cl_kernel g_subm_conv_fp16_lite_kernel = nullptr;
// NEW: Morton-ordered kernel
static cl_kernel g_subm_conv_fp16_morton_kernel = nullptr;
// NEW: Feature permutation kernel
static cl_kernel g_permute_features_fp16_kernel = nullptr;
// NEW: Transposed-weight kernels (better cache line utilization)
static cl_kernel g_subm_conv_fp16_w4t_kernel = nullptr;
static cl_kernel g_subm_conv_fp16_w8t_kernel = nullptr;

// Transposed weight buffers: [27, C_out/G, C_in, G] where G=4 or 8
static cl_mem g_layer_weights_t4[MAX_LAYERS] = {};  // for w4t
static cl_mem g_layer_weights_t8[MAX_LAYERS] = {};  // for w8t

// DPAS weight buffers (VNNI packed format)
static cl_mem g_layer_weights_dpas[MAX_LAYERS] = {};

// Morton ordering buffers (per level)
static cl_mem g_morton_order_buf[4] = {};
static size_t g_morton_order_sz[4] = {};
static bool g_use_dpas = false;
static bool g_use_morton = false;

// Fused sparse-to-BEV kernel
static cl_kernel g_sparse_to_bev_fp16_kernel = nullptr;
// FP16 output variant (half BEV readback bandwidth)
static cl_kernel g_sparse_to_bev_fp16_out_kernel = nullptr;

// Fused BN+ReLU+F32→F16 kernel
static cl_kernel g_fused_bn_relu_f16_kernel = nullptr;

// Compact hash table kernels (GPU neighbor map building)
static cl_kernel g_build_hash_compact_kernel = nullptr;
static cl_kernel g_build_nmap_compact_kernel = nullptr;

// NEW: Optimized lite8 (8-wide) and fused residual kernels
static cl_kernel g_subm_conv_fp16_lite8_kernel = nullptr;
static cl_kernel g_subm_conv_fp16_lite_resadd_kernel = nullptr;
static cl_kernel g_subm_conv_fp16_lite8_resadd_kernel = nullptr;
static cl_mem g_hash_keys_buf = nullptr;
static cl_mem g_hash_vals_buf = nullptr;
static size_t g_hk_sz = 0, g_hv_sz = 0;

static int g_call_count = 0;

#define CL_CHECK(call) do { \
    cl_int err_ = (call); \
    if (err_ != CL_SUCCESS) \
        throw std::runtime_error("OpenCL error " + std::to_string(err_) + \
            " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
} while(0)

static void ensure_buf(cl_mem& buf, size_t& cur, size_t need, cl_mem_flags flags) {
    if (need > cur) {
        if (buf) clReleaseMemObject(buf);
        cl_int e;
        buf = clCreateBuffer(g_context, flags, need, nullptr, &e);
        CL_CHECK(e);
        cur = need;
    }
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

static std::vector<uint16_t> to_fp16(const float* d, int n) {
    std::vector<uint16_t> out(n);
    for (int i = 0; i < n; i++) out[i] = f32_to_f16(d[i]);
    return out;
}

static void upload(cl_mem buf, const void* data, size_t sz) {
    CL_CHECK(clEnqueueWriteBuffer(g_queue, buf, CL_FALSE, 0, sz, data, 0, nullptr, nullptr));
}

static void gpu_f32_to_f16(cl_mem f32, cl_mem f16, int total) {
    CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 0, sizeof(cl_mem), &f32));
    CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 1, sizeof(cl_mem), &f16));
    CL_CHECK(clSetKernelArg(g_float_to_half_kernel, 2, sizeof(int), &total));
    size_t gws = (size_t)total;
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_float_to_half_kernel, 1, nullptr, &gws, nullptr, 0, nullptr, nullptr));
}

// ============================================================
// OpenCL Initialization
// ============================================================
static void init_opencl() {
    cl_int err;
    cl_uint npf;
    err = clGetPlatformIDs(0, nullptr, &npf);
    if (err != CL_SUCCESS || npf == 0)
        throw std::runtime_error("No OpenCL platforms");

    std::vector<cl_platform_id> plats(npf);
    clGetPlatformIDs(npf, plats.data(), nullptr);

    for (auto& p : plats) {
        cl_uint nd;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) continue;
        std::vector<cl_device_id> devs(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
        for (auto& d : devs) {
            char vendor[256];
            clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
            if (std::string(vendor).find("Intel") == std::string::npos) continue;
            g_device = d;
            g_context = clCreateContext(nullptr, 1, &g_device, nullptr, nullptr, &err);
            CL_CHECK(err);
            g_queue = clCreateCommandQueue(g_context, g_device, 0, &err);
            CL_CHECK(err);

            // Find kernel source (try multiple paths, including relative to this source file)
            std::string src;
            // Derive directory of this source file at compile time
            std::string this_dir = std::string(__FILE__);
            {
                auto pos = this_dir.find_last_of("/\\");
                if (pos != std::string::npos) this_dir = this_dir.substr(0, pos);
                else this_dir = ".";
            }
            for (const auto& path : {
                (this_dir + "/sparse_conv.cl"),
                // std::string("openvino_extensions/sparse_encoder/sparse_conv_fused.cl"),
                // std::string("bevfusion/kernels/sparse_conv/sparse_conv_fused.cl")
            }) {
                std::ifstream f(path);
                if (f.is_open()) {
                    std::ostringstream ss;
                    ss << f.rdbuf();
                    src = ss.str();
                    break;
                }
            }
            if (src.empty())
                throw std::runtime_error("Cannot find sparse_conv.cl");

            const char* sp = src.c_str();
            size_t sl = src.size();
            g_program = clCreateProgramWithSource(g_context, 1, &sp, &sl, &err);
            CL_CHECK(err);

            err = clBuildProgram(g_program, 1, &g_device, "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
            if (err != CL_SUCCESS) {
                char log[16384];
                size_t ll;
                clGetProgramBuildInfo(g_program, g_device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, &ll);
                throw std::runtime_error(std::string("Kernel build failed:\n") + log);
            }

            g_subm_conv_fp16_kernel        = clCreateKernel(g_program, "subm_conv_fp16", &err); CL_CHECK(err);
            g_subm_conv_fp16_scalar_kernel = clCreateKernel(g_program, "subm_conv_fp16_scalar", &err); CL_CHECK(err);
            g_subm_conv_fp16_w4_kernel     = clCreateKernel(g_program, "subm_conv_fp16_w4", &err); CL_CHECK(err);
            g_subm_conv_fp16_w16_kernel    = clCreateKernel(g_program, "subm_conv_fp16_w16", &err); CL_CHECK(err);
            g_subm_conv_fp16_slm_kernel    = clCreateKernel(g_program, "subm_conv_fp16_slm", &err);
            if (err != CL_SUCCESS) { g_subm_conv_fp16_slm_kernel = nullptr; }
            g_subm_conv_fp16_simd_kernel   = clCreateKernel(g_program, "subm_conv_fp16_simd", &err);
            if (err != CL_SUCCESS) { g_subm_conv_fp16_simd_kernel = nullptr; }
            g_subm_conv_fp16_local_kernel  = clCreateKernel(g_program, "subm_conv_fp16_local", &err);
            if (err != CL_SUCCESS) { g_subm_conv_fp16_local_kernel = nullptr; }
            g_residual_add_relu_fp16_kernel = clCreateKernel(g_program, "residual_add_relu_fp16", &err); CL_CHECK(err);
            g_strided_conv_fp16_kernel     = clCreateKernel(g_program, "strided_conv_scatter_fp16", &err); CL_CHECK(err);
            g_strided_conv_fp16_compact_kernel = clCreateKernel(g_program, "strided_conv_scatter_fp16_compact", &err); CL_CHECK(err);
            g_strided_conv_gather_fp16_kernel = clCreateKernel(g_program, "strided_conv_gather_fp16", &err); CL_CHECK(err);
            g_apply_bn_relu_kernel         = clCreateKernel(g_program, "apply_bn_relu", &err); CL_CHECK(err);
            g_float_to_half_kernel         = clCreateKernel(g_program, "float_to_half_kernel", &err); CL_CHECK(err);
            g_sparse_to_dense_fp16_kernel  = clCreateKernel(g_program, "sparse_to_dense_fp16", &err); CL_CHECK(err);

            // Optional fused kernel (write directly to BEV layout)
            g_sparse_to_bev_fp16_kernel = clCreateKernel(g_program, "sparse_to_bev_fp16", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] sparse_to_bev_fp16 not found, using fallback" << std::endl;
                g_sparse_to_bev_fp16_kernel = nullptr;
            }
            // FP16 output variant (half the BEV readback bandwidth)
            g_sparse_to_bev_fp16_out_kernel = clCreateKernel(g_program, "sparse_to_bev_fp16_out", &err);
            if (err != CL_SUCCESS) {
                g_sparse_to_bev_fp16_out_kernel = nullptr;
            }

            // Fused BN+ReLU+F32→F16 kernel
            g_fused_bn_relu_f16_kernel = clCreateKernel(g_program, "fused_bn_relu_f16", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] fused_bn_relu_f16 not found, using fallback" << std::endl;
                g_fused_bn_relu_f16_kernel = nullptr;
            }

            // Compact GPU hash table kernels (for GPU neighbor map building)
            g_build_hash_compact_kernel = clCreateKernel(g_program, "build_hash_table_compact", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] build_hash_table_compact not found" << std::endl;
                g_build_hash_compact_kernel = nullptr;
            }
            g_build_nmap_compact_kernel = clCreateKernel(g_program, "build_neighbor_map_compact", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] build_neighbor_map_compact not found" << std::endl;
                g_build_nmap_compact_kernel = nullptr;
            }

            // SLM v2 kernels
            g_subm_conv_fp16_slm_v2_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_v2", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_slm_v2 not found" << std::endl;
                g_subm_conv_fp16_slm_v2_kernel = nullptr;
            }
            g_subm_conv_fp16_slm_v2_w8_kernel = clCreateKernel(g_program, "subm_conv_fp16_slm_v2_w8", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_slm_v2_w8 not found" << std::endl;
                g_subm_conv_fp16_slm_v2_w8_kernel = nullptr;
            }
            g_subm_conv_fp16_mv4_kernel = clCreateKernel(g_program, "subm_conv_fp16_mv4", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_mv4 not found" << std::endl;
                g_subm_conv_fp16_mv4_kernel = nullptr;
            }

            // Strided conv with precomputed neighbor map
            g_build_strided_nmap_kernel = clCreateKernel(g_program, "build_strided_nmap", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] build_strided_nmap not found" << std::endl;
                g_build_strided_nmap_kernel = nullptr;
            }
            g_strided_conv_nmap_fp16_kernel = clCreateKernel(g_program, "strided_conv_nmap_fp16", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] strided_conv_nmap_fp16 not found" << std::endl;
                g_strided_conv_nmap_fp16_kernel = nullptr;
            }

            // NEW: DPAS/XMX kernels
            g_subm_conv_fp16_dpas_kernel = clCreateKernel(g_program, "subm_conv_fp16_dpas", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_dpas not found (XMX not available)" << std::endl;
                g_subm_conv_fp16_dpas_kernel = nullptr;
            } else {
                // DPAS kernel compiled but may not work on all GPUs
                // Verified: 0xb0b0 (32 EU Xe-LPG) reports XMX but lacks functional units
                // Disable DPAS until we can verify correctness at runtime
                g_use_dpas = false;
                std::cout << "[sparse_encoder] DPAS kernel compiled (disabled: verification needed)" << std::endl;
            }
            g_subm_conv_fp16_dpas_resadd_kernel = clCreateKernel(g_program, "subm_conv_fp16_dpas_resadd", &err);
            if (err != CL_SUCCESS) {
                g_subm_conv_fp16_dpas_resadd_kernel = nullptr;
            }

            // NEW: Optimized FP16 fast kernel
            g_subm_conv_fp16_fast_kernel = clCreateKernel(g_program, "subm_conv_fp16_fast", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_fast not found" << std::endl;
                g_subm_conv_fp16_fast_kernel = nullptr;
            }

            // NEW: FP16 lite kernel (FP16 accum, no preloading)
            g_subm_conv_fp16_lite_kernel = clCreateKernel(g_program, "subm_conv_fp16_lite", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_lite not found" << std::endl;
                g_subm_conv_fp16_lite_kernel = nullptr;
            }

            // NEW: Morton-ordered kernel
            g_subm_conv_fp16_morton_kernel = clCreateKernel(g_program, "subm_conv_fp16_morton", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_morton not found" << std::endl;
                g_subm_conv_fp16_morton_kernel = nullptr;
            }

            // NEW: Feature permutation kernel
            g_permute_features_fp16_kernel = clCreateKernel(g_program, "permute_features_fp16", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] permute_features_fp16 not found" << std::endl;
                g_permute_features_fp16_kernel = nullptr;
            }

            // NEW: Transposed-weight kernels
            g_subm_conv_fp16_w4t_kernel = clCreateKernel(g_program, "subm_conv_fp16_w4t", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_w4t not found" << std::endl;
                g_subm_conv_fp16_w4t_kernel = nullptr;
            }
            g_subm_conv_fp16_w8t_kernel = clCreateKernel(g_program, "subm_conv_fp16_w8t", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_w8t not found" << std::endl;
                g_subm_conv_fp16_w8t_kernel = nullptr;
            }

            // NEW: Optimized lite8 and fused residual kernels
            g_subm_conv_fp16_lite8_kernel = clCreateKernel(g_program, "subm_conv_fp16_lite8", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_lite8 not found" << std::endl;
                g_subm_conv_fp16_lite8_kernel = nullptr;
            }
            g_subm_conv_fp16_lite_resadd_kernel = clCreateKernel(g_program, "subm_conv_fp16_lite_resadd", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_lite_resadd not found" << std::endl;
                g_subm_conv_fp16_lite_resadd_kernel = nullptr;
            }
            g_subm_conv_fp16_lite8_resadd_kernel = clCreateKernel(g_program, "subm_conv_fp16_lite8_resadd", &err);
            if (err != CL_SUCCESS) {
                std::cout << "[sparse_encoder] subm_conv_fp16_lite8_resadd not found" << std::endl;
                g_subm_conv_fp16_lite8_resadd_kernel = nullptr;
            }

            char name[256];
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            std::cout << "[sparse_encoder_ext] Using: " << name << std::endl;
            return;
        }
    }
    throw std::runtime_error("No Intel GPU found");
}


// ============================================================
// CPU Neighbor Map (open-address hash, same as original)
// ============================================================
struct FlatHash {
    std::vector<int64_t> keys;
    std::vector<int> vals;
    int mask;

    FlatHash(int n) {
        int cap = 1;
        while (cap < n * 2) cap *= 2;
        mask = cap - 1;
        keys.assign(cap, -1LL);
        vals.resize(cap, -1);
    }
    void insert(int64_t k, int v) {
        int h = (int)((uint64_t)k * 2654435761ULL >> 32) & mask;
        while (keys[h] != -1LL) h = (h + 1) & mask;
        keys[h] = k; vals[h] = v;
    }
    int find(int64_t k) const {
        int h = (int)((uint64_t)k * 2654435761ULL >> 32) & mask;
        while (keys[h] != -1LL) {
            if (keys[h] == k) return vals[h];
            h = (h + 1) & mask;
        }
        return -1;
    }
};

static std::vector<int> build_nmap(const int* coords, int N, int X, int Y, int Z) {
    FlatHash ht(N);
    for (int i = 0; i < N; i++) {
        int64_t key = ((int64_t)coords[i*4+1] * Y + coords[i*4+2]) * Z + coords[i*4+3];
        ht.insert(key, i);
    }
    std::vector<int> nmap(N * 27, -1);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < N; i++) {
        int x = coords[i*4+1], y = coords[i*4+2], z = coords[i*4+3];
        int k = 0;
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
                for (int dz = -1; dz <= 1; dz++) {
                    int nx = x+dx, ny = y+dy, nz = z+dz;
                    if (nx >= 0 && nx < X && ny >= 0 && ny < Y && nz >= 0 && nz < Z) {
                        int64_t key = ((int64_t)nx * Y + ny) * Z + nz;
                        nmap[i*27+k] = ht.find(key);
                    }
                    k++;
                }
    }
    return nmap;
}

// ============================================================
// GPU Neighbor Map Building (compact hash table)
// Builds compact hash table on GPU.
// After this call, g_hash_keys_buf and g_hash_vals_buf contain the hash table.
// Returns table_mask for use in hash lookups.
// ============================================================
static int g_last_table_mask = 0;

static int gpu_build_hash_table(cl_mem coords_buf, int N, int Y, int Z) {
    if (!g_build_hash_compact_kernel) {
        throw std::runtime_error("GPU hash kernel not available");
    }
    
    // Table size: next power of 2 >= 2*N
    int table_size = 1;
    while (table_size < N * 2) table_size *= 2;
    int table_mask = table_size - 1;
    g_last_table_mask = table_mask;
    
    // Allocate hash table buffers
    size_t hb = (size_t)table_size * sizeof(int);
    ensure_buf(g_hash_keys_buf, g_hk_sz, hb, CL_MEM_READ_WRITE);
    ensure_buf(g_hash_vals_buf, g_hv_sz, hb, CL_MEM_READ_WRITE);
    
    // Initialize keys to -1
    int neg1 = -1;
    CL_CHECK(clEnqueueFillBuffer(g_queue, g_hash_keys_buf, &neg1, sizeof(int), 0, hb, 0, nullptr, nullptr));
    
    // Build hash table
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 0, sizeof(cl_mem), &coords_buf));
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 1, sizeof(cl_mem), &g_hash_keys_buf));
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 2, sizeof(cl_mem), &g_hash_vals_buf));
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 3, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 4, sizeof(int), &Y));
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 5, sizeof(int), &Z));
    CL_CHECK(clSetKernelArg(g_build_hash_compact_kernel, 6, sizeof(int), &table_mask));
    size_t gws_ht = (size_t)N;
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_build_hash_compact_kernel,
                                     1, nullptr, &gws_ht, nullptr, 0, nullptr, nullptr));
    return table_mask;
}

// Builds neighbor map using existing hash table in g_hash_keys_buf/g_hash_vals_buf.
// ============================================================
static void gpu_build_nmap_from_hash(cl_mem coords_buf, cl_mem nmap_buf,
                                      int N, int X, int Y, int Z, int table_mask) {
    if (!g_build_nmap_compact_kernel) {
        throw std::runtime_error("GPU nmap kernel not available");
    }
    
    // Initialize nmap to -1
    size_t nb = (size_t)N * 27 * sizeof(int);
    int neg1 = -1;
    CL_CHECK(clEnqueueFillBuffer(g_queue, nmap_buf, &neg1, sizeof(int), 0, nb, 0, nullptr, nullptr));
    
    // Build neighbor map
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 0, sizeof(cl_mem), &coords_buf));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 1, sizeof(cl_mem), &g_hash_keys_buf));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 2, sizeof(cl_mem), &g_hash_vals_buf));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 3, sizeof(cl_mem), &nmap_buf));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 4, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 5, sizeof(int), &X));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 6, sizeof(int), &Y));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 7, sizeof(int), &Z));
    CL_CHECK(clSetKernelArg(g_build_nmap_compact_kernel, 8, sizeof(int), &table_mask));
    size_t gws_nm = (size_t)N;
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_build_nmap_compact_kernel,
                                     1, nullptr, &gws_nm, nullptr, 0, nullptr, nullptr));
}

// Combined: build hash + nmap
static void gpu_build_nmap(cl_mem coords_buf, cl_mem nmap_buf,
                           int N, int X, int Y, int Z) {
    int table_mask = gpu_build_hash_table(coords_buf, N, Y, Z);
    gpu_build_nmap_from_hash(coords_buf, nmap_buf, N, X, Y, Z, table_mask);
}

// ============================================================
// Layer Descriptor (hardcoded BEVFusion architecture)
// ============================================================
struct LayerDesc {
    int type;     // 0=subm_conv, 1=residual_block_start, 2=downsample, 3=conv_out
    int C_in, C_out;
    int num_k;    // number of kernel elements (27 for 3×3×3, 3 for 1×1×3)
    int kx, ky, kz, sx, sy, sz, px, py, pz;
};

static const LayerDesc LAYERS[] = {
    // conv_input: SubM 5→16
    {0, 5, 16, 27,  3,3,3, 1,1,1, 0,0,0},
    // encoder_layer0: 2 res blocks + downsample
    {1, 16, 16, 27,  3,3,3, 1,1,1, 0,0,0},  // res0_0_c1
    {1, 16, 16, 27,  3,3,3, 1,1,1, 0,0,0},  // res0_0_c2
    {1, 16, 16, 27,  3,3,3, 1,1,1, 0,0,0},  // res0_1_c1
    {1, 16, 16, 27,  3,3,3, 1,1,1, 0,0,0},  // res0_1_c2
    {2, 16, 32, 27,  3,3,3, 2,2,2, 1,1,1},  // ds0
    // encoder_layer1
    {1, 32, 32, 27,  3,3,3, 1,1,1, 0,0,0},
    {1, 32, 32, 27,  3,3,3, 1,1,1, 0,0,0},
    {1, 32, 32, 27,  3,3,3, 1,1,1, 0,0,0},
    {1, 32, 32, 27,  3,3,3, 1,1,1, 0,0,0},
    {2, 32, 64, 27,  3,3,3, 2,2,2, 1,1,1},  // ds1
    // encoder_layer2
    {1, 64, 64,  27, 3,3,3, 1,1,1, 0,0,0},
    {1, 64, 64,  27, 3,3,3, 1,1,1, 0,0,0},
    {1, 64, 64,  27, 3,3,3, 1,1,1, 0,0,0},
    {1, 64, 64,  27, 3,3,3, 1,1,1, 0,0,0},
    {2, 64, 128, 27, 3,3,3, 2,2,2, 1,1,1},  // ds2
    // encoder_layer3 (no downsample)
    {1, 128, 128, 27, 3,3,3, 1,1,1, 0,0,0},
    {1, 128, 128, 27, 3,3,3, 1,1,1, 0,0,0},
    {1, 128, 128, 27, 3,3,3, 1,1,1, 0,0,0},
    {1, 128, 128, 27, 3,3,3, 1,1,1, 0,0,0},
    // conv_out: 128→128, k=1×1×3, s=1×1×2, p=0
    {3, 128, 128, 3,  1,1,3, 1,1,2, 0,0,0},
};
constexpr int NUM_LAYERS = 21;

// Persistent GPU buffers for pre-computed neighbor maps at each resolution level
// Level 0: input coords (1440x1440x41)
// Level 1: after ds0 (720x720x21)
// Level 2: after ds1 (360x360x11)
// Level 3: after ds2 (180x180x5) -- conv_out uses these coords, no nmap needed
constexpr int MAX_LEVELS = 4;
static cl_mem g_level_nmap_buf[MAX_LEVELS] = {};
static cl_mem g_level_coords_buf[MAX_LEVELS] = {};
static size_t g_level_nmap_sz[MAX_LEVELS] = {};
static size_t g_level_coords_sz[MAX_LEVELS] = {};

// Strided conv precomputed neighbor maps (GPU buffers)
static cl_mem g_strided_nmap_buf[4] = {};
static size_t g_strided_nmap_sz[4] = {};

// Pre-computed strided conv data
struct StridedConvPrecomp {
    std::vector<int> out_coords;
    std::vector<int> morton_perm;  // Morton sort permutation: sorted[i] = original[perm[i]]
    int N_out;
    int X_out, Y_out, Z_out;
};

// Pre-compute ALL strided conv coordinate maps from input coordinates
// This is purely geometric — no dependence on feature values
static void precompute_all_coord_maps(
    const int* input_coords, int N,
    int X0, int Y0, int Z0,
    // outputs:
    std::vector<std::vector<int>>& level_nmaps,      // [4] neighbor maps
    std::vector<std::vector<int>>& level_coords,     // [4] coordinate arrays  
    std::vector<int>& level_N,                       // [4] voxel counts
    std::vector<int>& level_X,                       // [4] grid dims
    std::vector<int>& level_Y,
    std::vector<int>& level_Z,
    StridedConvPrecomp strided_precomp[4]             // data for each strided conv
) {
    // Level 0: input coordinates
    level_coords[0].assign(input_coords, input_coords + N * 4);
    level_N[0] = N;
    level_X[0] = X0; level_Y[0] = Y0; level_Z[0] = Z0;
    level_nmaps[0] = build_nmap(input_coords, N, X0, Y0, Z0);

    // Iterate through strided convs to generate subsequent levels
    const int strided_layers[] = {5, 10, 15, 20};  // layer indices of strided convs
    int cur_X = X0, cur_Y = Y0, cur_Z = Z0;
    
    for (int s = 0; s < 4; s++) {
        int li = strided_layers[s];
        const auto& L = LAYERS[li];
        int kx = L.kx, ky = L.ky, kz = L.kz;
        int sx = L.sx, sy = L.sy, sz = L.sz;
        int px = L.px, py = L.py, pz = L.pz;

        int X_out = (cur_X + 2*px - kx) / sx + 1;
        int Y_out = (cur_Y + 2*py - ky) / sy + 1;
        int Z_out = (cur_Z + 2*pz - kz) / sz + 1;

        const auto& src_coords = level_coords[s];
        int src_N = level_N[s];

        size_t oh_entries = (size_t)X_out * Y_out * Z_out;
        std::vector<int> h_out_hash(oh_entries, -1);
        std::vector<int> out_coords;
        out_coords.reserve(src_N * 4);
        int next_idx = 0;

        for (int i = 0; i < src_N; i++) {
            int batch = src_coords[i*4+0];
            int in_x = src_coords[i*4+1], in_y = src_coords[i*4+2], in_z = src_coords[i*4+3];
            for (int ki = 0; ki < kx; ki++)
                for (int kj = 0; kj < ky; kj++)
                    for (int kk = 0; kk < kz; kk++) {
                        int ox = in_x + px - ki, oy = in_y + py - kj, oz = in_z + pz - kk;
                        if (ox >= 0 && ox % sx == 0 && oy >= 0 && oy % sy == 0 &&
                            oz >= 0 && oz % sz == 0) {
                            int out_x = ox/sx, out_y = oy/sy, out_z = oz/sz;
                            if (out_x >= 0 && out_x < X_out &&
                                out_y >= 0 && out_y < Y_out &&
                                out_z >= 0 && out_z < Z_out) {
                                int64_t key = ((int64_t)out_x * Y_out + out_y) * Z_out + out_z;
                                if (h_out_hash[key] == -1) {
                                    h_out_hash[key] = next_idx++;
                                    out_coords.push_back(batch);
                                    out_coords.push_back(out_x);
                                    out_coords.push_back(out_y);
                                    out_coords.push_back(out_z);
                                }
                            }
                        }
                    }
        }
        int N_out = (int)out_coords.size() / 4;

        // Store strided conv precomputed data
        strided_precomp[s].out_coords = out_coords;
        strided_precomp[s].N_out = N_out;
        strided_precomp[s].X_out = X_out;
        strided_precomp[s].Y_out = Y_out;
        strided_precomp[s].Z_out = Z_out;

        // Next level coords (skip nmap for conv_out, level 3)
        if (s < 3) {
            level_coords[s+1] = out_coords;
            level_N[s+1] = N_out;
            level_X[s+1] = X_out; level_Y[s+1] = Y_out; level_Z[s+1] = Z_out;
            level_nmaps[s+1] = build_nmap(out_coords.data(), N_out, X_out, Y_out, Z_out);
        }

        cur_X = X_out; cur_Y = Y_out; cur_Z = Z_out;
    }
}


// ============================================================
// SubM Conv GPU dispatch (FP16) — uses pre-uploaded weights
// Auto-selects between w8, w4, scalar, w16, SLM variants based
// on benchmark results (determined once per channel configuration).
// ============================================================
// Kernel selection state: index into variants array, -1 = not benchmarked
static int g_best_kernel[4] = {-1, -1, -1, -1}; // per channel width: 16,32,64,128
// 0=standard, 1=slm_1d(slm_v2/slm_v2_w8), 2=slm_1d(old slm kernel), 3=local_2d, 4=mv4
static int g_best_kernel_type[4] = {};
static int g_best_kernel_idx[4] = {};

static void dispatch_subm_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                 int N, int Cin, int Cout, int relu,
                                 int layer_idx, int dim0_divisor) {
    int wg_dim0 = (Cout + dim0_divisor - 1) / dim0_divisor;
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    size_t gws[2] = {(size_t)wg_dim0, (size_t)N};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
}

// Dispatch SLM-based kernel (1D, one WG per voxel)
static void dispatch_slm_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                int N, int Cin, int Cout, int relu,
                                int layer_idx, int channels_per_wi) {
    int WG = Cout / channels_per_wi;
    size_t slm_bytes = (size_t)27 * Cin * sizeof(uint16_t);
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes, nullptr));
    size_t gws1 = (size_t)N * WG;
    size_t lws1 = (size_t)WG;
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &gws1, &lws1, 0, nullptr, nullptr));
}

// Dispatch multi-voxel SLM kernel (MV_NV=4 voxels per WG)
static void dispatch_mv4_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                int N, int Cin, int Cout, int relu,
                                int layer_idx) {
    int vox_per_wg = 4;
    int wis_per_voxel = Cout / 4;
    int WG = vox_per_wg * wis_per_voxel;
    int num_groups = (N + vox_per_wg - 1) / vox_per_wg;
    size_t slm_bytes = (size_t)vox_per_wg * 27 * Cin * sizeof(uint16_t);
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes, nullptr));
    size_t gws1 = (size_t)num_groups * WG;
    size_t lws1 = (size_t)WG;
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 1, nullptr, &gws1, &lws1, 0, nullptr, nullptr));
}

// Dispatch local (2D) SLM kernel
static void dispatch_local_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                  int N, int Cin, int Cout, int relu,
                                  int layer_idx, int channels_per_wi) {
    int WG = Cout / channels_per_wi;
    size_t slm_bytes = (size_t)27 * Cin * sizeof(uint16_t);
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    CL_CHECK(clSetKernelArg(kern, 10, slm_bytes, nullptr));
    size_t gws2[2] = {(size_t)N, (size_t)WG};
    size_t lws2[2] = {1, (size_t)WG};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws2, lws2, 0, nullptr, nullptr));
}

// ── DPAS dispatch: 16 voxels × 8 output channels per sub-group ──
static void dispatch_dpas_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                 int N, int Cin, int Cout, int relu,
                                 int layer_idx) {
    if (!g_layer_weights_dpas[layer_idx]) return;
    int C_out_tiles = Cout / 8;
    int N_padded = ((N + 15) / 16) * 16;  // pad to multiple of 16
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights_dpas[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    size_t gws[2] = {(size_t)C_out_tiles, (size_t)N_padded};
    size_t lws[2] = {1, 16};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, lws, 0, nullptr, nullptr));
}

// ── DPAS dispatch with residual add ──
static void dispatch_dpas_resadd_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                        cl_mem identity, int N, int Cin, int Cout,
                                        int relu, int layer_idx) {
    if (!g_layer_weights_dpas[layer_idx]) return;
    int C_out_tiles = Cout / 8;
    int N_padded = ((N + 15) / 16) * 16;
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights_dpas[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(cl_mem), &identity));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 10, sizeof(int), &relu));
    size_t gws[2] = {(size_t)C_out_tiles, (size_t)N_padded};
    size_t lws[2] = {1, 16};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, lws, 0, nullptr, nullptr));
}

// ── Fast dispatch (FP16 acc + prefetch + neighbor preload) ──
static void dispatch_fast_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                 int N, int Cin, int Cout, int relu,
                                 int layer_idx) {
    int wg_dim0 = (Cout + 3) / 4;
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    size_t gws[2] = {(size_t)wg_dim0, (size_t)N};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
}

// Dispatch transposed-weight w4t kernel
static void dispatch_w4t_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                int N, int Cin, int Cout, int relu,
                                int layer_idx) {
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights_t4[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    int wg_dim0 = Cout / 4;
    size_t gws[2] = {(size_t)wg_dim0, (size_t)N};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
}

// Dispatch fused conv+resadd kernel (lite4 or lite8 variant)
static void dispatch_resadd_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                   cl_mem identity, int N, int Cin, int Cout,
                                   int layer_idx, int dim0_divisor) {
    int wg_dim0 = (Cout + dim0_divisor - 1) / dim0_divisor;
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(cl_mem), &identity));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &Cout));
    size_t gws[2] = {(size_t)wg_dim0, (size_t)N};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
}

// Dispatch transposed-weight w8t kernel
static void dispatch_w8t_kernel(cl_kernel kern, cl_mem fin, cl_mem fout,
                                int N, int Cin, int Cout, int relu,
                                int layer_idx) {
    CL_CHECK(clSetKernelArg(kern, 0, sizeof(cl_mem), &fin));
    CL_CHECK(clSetKernelArg(kern, 1, sizeof(cl_mem), &g_neighbor_map_buf));
    CL_CHECK(clSetKernelArg(kern, 2, sizeof(cl_mem), &g_layer_weights_t8[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 3, sizeof(cl_mem), &g_layer_scales[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 4, sizeof(cl_mem), &g_layer_biases[layer_idx]));
    CL_CHECK(clSetKernelArg(kern, 5, sizeof(cl_mem), &fout));
    CL_CHECK(clSetKernelArg(kern, 6, sizeof(int), &N));
    CL_CHECK(clSetKernelArg(kern, 7, sizeof(int), &Cin));
    CL_CHECK(clSetKernelArg(kern, 8, sizeof(int), &Cout));
    CL_CHECK(clSetKernelArg(kern, 9, sizeof(int), &relu));
    int wg_dim0 = Cout / 8;
    size_t gws[2] = {(size_t)wg_dim0, (size_t)N};
    CL_CHECK(clEnqueueNDRangeKernel(g_queue, kern, 2, nullptr, gws, nullptr, 0, nullptr, nullptr));
}

static void run_subm_conv(
    cl_mem fin, cl_mem fout,
    int N, int Cin, int Cout, int relu,
    int layer_idx
) {
    // Channel width index: 16→0, 32→1, 64→2, 128→3
    int cw_idx = -1;
    if (Cin == 16 && Cout == 16) cw_idx = 0;
    else if (Cin == 32 && Cout == 32) cw_idx = 1;
    else if (Cin == 64 && Cout == 64) cw_idx = 2;
    else if (Cin == 128 && Cout == 128) cw_idx = 3;

    // Benchmark on first call for each channel config
    if (cw_idx >= 0 && g_best_kernel[cw_idx] < 0) {
        double best_ms = 1e30;
        int best_type = 0, best_idx = 0;
        const int WARMUP = 2, RUNS = 4;

        auto bench = [&](const char* name, auto dispatch_fn) {
            try {
                for (int r = 0; r < WARMUP; r++) dispatch_fn();
                CL_CHECK(clFinish(g_queue));
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int r = 0; r < RUNS; r++) dispatch_fn();
                CL_CHECK(clFinish(g_queue));
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t0).count() / RUNS;
                std::cout << "[subm_bench] " << Cin << "x" << Cout << " N=" << N
                          << " " << name << ": " << ms << " ms" << std::endl;
                return ms;
            } catch (const std::exception& e) {
                std::cout << "[subm_bench] " << name << " FAILED: " << e.what() << std::endl;
                return 1e30;
            }
        };

        // Standard variants (2D dispatch, no SLM)
        struct { cl_kernel k; int div; const char* name; } std_v[] = {
            {g_subm_conv_fp16_kernel, 8, "w8"},
            {g_subm_conv_fp16_w4_kernel, 4, "w4"},
            {g_subm_conv_fp16_lite_kernel, 4, "lite"},
            {g_subm_conv_fp16_lite8_kernel, 8, "lite8"},
        };
        int n_std = 4;
        for (int v = 0; v < n_std; v++) {
            if (!std_v[v].k) continue;
            double ms = bench(std_v[v].name, [&](){
                dispatch_subm_kernel(std_v[v].k, fin, fout, N, Cin, Cout, relu, layer_idx, std_v[v].div);
            });
            if (ms < best_ms) { best_ms = ms; best_type = 0; best_idx = v; }
        }

        // ── NEW: Fast kernel (FP16 acc + prefetch + neighbor preload) ──
        if (g_subm_conv_fp16_fast_kernel) {
            double ms = bench("fast", [&](){
                dispatch_fast_kernel(g_subm_conv_fp16_fast_kernel, fin, fout,
                                    N, Cin, Cout, relu, layer_idx);
            });
            if (ms < best_ms) { best_ms = ms; best_type = 6; best_idx = 0; }
        }

        g_best_kernel[cw_idx] = 1; // mark as benchmarked
        g_best_kernel_type[cw_idx] = best_type;
        g_best_kernel_idx[cw_idx] = best_idx;
        std::cout << "[subm_bench] BEST for " << Cin << "x" << Cout
                  << ": type=" << best_type << " idx=" << best_idx
                  << " (" << best_ms << " ms)" << std::endl;
    }

    // Dispatch using best kernel
    if (cw_idx >= 0 && g_best_kernel[cw_idx] > 0) {
        int type = g_best_kernel_type[cw_idx];
        int idx = g_best_kernel_idx[cw_idx];
        switch (type) {
            case 0: {
                cl_kernel kerns[] = {g_subm_conv_fp16_kernel, g_subm_conv_fp16_w4_kernel,
                                     g_subm_conv_fp16_lite_kernel, g_subm_conv_fp16_lite8_kernel,
                                     g_subm_conv_fp16_scalar_kernel, g_subm_conv_fp16_w16_kernel};
                int divs[] = {8, 4, 4, 8, 1, 16};
                dispatch_subm_kernel(kerns[idx], fin, fout, N, Cin, Cout, relu, layer_idx, divs[idx]);
                break;
            }
            case 1: // slm_v2 
                if (idx == 4) // w4
                    dispatch_slm_kernel(g_subm_conv_fp16_slm_v2_kernel, fin, fout,
                                       N, Cin, Cout, relu, layer_idx, 4);
                else // w8
                    dispatch_slm_kernel(g_subm_conv_fp16_slm_v2_w8_kernel, fin, fout,
                                       N, Cin, Cout, relu, layer_idx, 8);
                break;
            case 2: // old SLM
                dispatch_slm_kernel(g_subm_conv_fp16_slm_kernel, fin, fout,
                                   N, Cin, Cout, relu, layer_idx, 8);
                break;
            case 3: // local
                dispatch_local_kernel(g_subm_conv_fp16_local_kernel, fin, fout,
                                     N, Cin, Cout, relu, layer_idx, 4);
                break;
            case 4: // mv4
                dispatch_mv4_kernel(g_subm_conv_fp16_mv4_kernel, fin, fout,
                                   N, Cin, Cout, relu, layer_idx);
                break;
            case 5: // DPAS/XMX
                dispatch_dpas_kernel(g_subm_conv_fp16_dpas_kernel, fin, fout,
                                   N, Cin, Cout, relu, layer_idx);
                break;
            case 6: // fast (FP16 acc + prefetch)
                dispatch_fast_kernel(g_subm_conv_fp16_fast_kernel, fin, fout,
                                   N, Cin, Cout, relu, layer_idx);
                break;
            case 7: // transposed weights (w4t, w8t)
                if (idx == 0)
                    dispatch_w4t_kernel(g_subm_conv_fp16_w4t_kernel, fin, fout,
                                       N, Cin, Cout, relu, layer_idx);
                else
                    dispatch_w8t_kernel(g_subm_conv_fp16_w8t_kernel, fin, fout,
                                       N, Cin, Cout, relu, layer_idx);
                break;
        }
    } else {
        // Default: use w4 kernel for non-standard configs
        dispatch_subm_kernel(g_subm_conv_fp16_w4_kernel, fin, fout, N, Cin, Cout, relu, layer_idx, 4);
    }
}


// ============================================================
// OV Op boilerplate
// ============================================================
SparseEncoderOp::SparseEncoderOp(
    const ov::Output<ov::Node>& features,
    const ov::Output<ov::Node>& coords,
    const ov::Output<ov::Node>& num_voxels,
    const ov::Output<ov::Node>& params)
    : Op({features, coords, num_voxels, params}) {
    constructor_validate_and_infer_types();
}

void SparseEncoderOp::validate_and_infer_types() {
    set_output_type(0, ov::element::f32,
                    ov::Shape{1, SPENC_BEV_C, SPENC_BEV_H, SPENC_BEV_W});
}

std::shared_ptr<ov::Node> SparseEncoderOp::clone_with_new_inputs(
    const ov::OutputVector& args) const {
    return std::make_shared<SparseEncoderOp>(args[0], args[1], args[2], args[3]);
}

bool SparseEncoderOp::visit_attributes(ov::AttributeVisitor&) { return true; }
bool SparseEncoderOp::has_evaluate() const { return true; }


// ============================================================
// Pre-upload weights for all layers (one-time, persistent GPU buffers)
// Weight layout: [num_k, C_in, C_out] (original, optimal for SIMD coalescing)
// ============================================================
static void pre_upload_weights(const float* params, const int* w_offsets, const int* s_offsets, const int* b_offsets) {
    if (g_weights_uploaded) return;
    cl_int clerr;
    for (int i = 0; i < NUM_LAYERS; i++) {
        int wcount = LAYERS[i].num_k * LAYERS[i].C_in * LAYERS[i].C_out;
        auto fp16w = to_fp16(params + w_offsets[i], wcount);
        size_t wb = fp16w.size() * sizeof(uint16_t);
        g_layer_weights[i] = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             wb, fp16w.data(), &clerr);
        CL_CHECK(clerr);

        int Cout = LAYERS[i].C_out;
        size_t sb = (size_t)Cout * sizeof(float);
        g_layer_scales[i] = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sb, (void*)(params + s_offsets[i]), &clerr);
        CL_CHECK(clerr);
        g_layer_biases[i] = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            sb, (void*)(params + b_offsets[i]), &clerr);
        CL_CHECK(clerr);
    }
    g_weights_uploaded = true;
    std::cout << "[sparse_encoder_ext] Pre-uploaded weights for " << NUM_LAYERS << " layers" << std::endl;

    // ── DPAS weight packing: convert [num_k, C_in, C_out] to VNNI format ──
    // VNNI for FP16 k16: weights_packed[k][ci_tile][co_tile][8][16] as uint
    // Where the [8][16] block is loaded by intel_sub_group_block_read8
    // The block layout: for sub_group_size=16, block_read8 reads
    //   result[elem] = ptr[sg_lid + elem * 16]  for elem=0..7, sg_lid=0..15
    // So memory layout is: interleaved 16-wide rows, 8 rows
    // For a B tile [K=16, N=8] in half:
    //   We need 128 halfs = 64 uints packed as 8 rows × 16 cols (uints)
    //   Row r of block_read (r=0..7): ptr[sg_lid + r*16] for sg_lid=0..15
    //   Each uint = 2 halfs: one pair of K values for a given N column
    //   Layout: packed[r * 16 + sg_lid] = pack(B[r*2][sg_lid % 8], B[r*2+1][sg_lid % 8])
    //   But sg_lid goes 0..15 and N=8, so sg_lid 0..7 are columns 0..7, sg_lid 8..15 repeat
    //   Actually: for N=8, the block_read accesses words at offsets 0..15, 16..31, etc.
    //   We need a dense 128-uint block where:
    //     data[row * 16 + col] for row=0..7, col=0..15
    //   WI with sg_lid reads: data[sg_lid], data[sg_lid+16], ..., data[sg_lid+112]
    //   which is: data[0..15], data[16..31], ..., data[112..127]
    //   = columns of our 8×16 matrix
    if (g_use_dpas) {
        std::cout << "[sparse_encoder_ext] Packing weights for DPAS..." << std::endl;
        for (int i = 0; i < NUM_LAYERS; i++) {
            int num_k = LAYERS[i].num_k;
            int C_in = LAYERS[i].C_in;
            int C_out = LAYERS[i].C_out;
            
            // Only pack layers where C_in and C_out are multiples of 16 and 8
            if (C_in % 16 != 0 || C_out % 8 != 0) {
                g_layer_weights_dpas[i] = nullptr;
                continue;
            }
            
            int C_in_tiles = C_in / 16;
            int C_out_tiles = C_out / 8;
            
            // Read original FP16 weights
            int wcount = num_k * C_in * C_out;
            auto fp16w = to_fp16(params + w_offsets[i], wcount);
            const uint16_t* src = fp16w.data();
            
            // Original layout: [num_k, C_in, C_out]
            // w(k, ci, co) = src[k * C_in * C_out + ci * C_out + co]
            
            // Packed layout: [num_k, C_in_tiles, C_out_tiles, 128] as uint32
            // Each 128-uint block corresponds to one [K=16, N=8] tile
            // Within the block: 8 rows × 16 cols of uint32
            // row r, col c: packed[r * 16 + c]
            // Each uint32 packs 2 halfs from the K dimension
            // packed[r][c] = {B_half[2*r][c % 8], B_half[2*r+1][c % 8]}
            // where c wraps around (c % 8) for N=8, cols 8..15 duplicate 0..7
            
            size_t packed_count = (size_t)num_k * C_in_tiles * C_out_tiles * 128;
            std::vector<uint32_t> packed(packed_count, 0);
            
            for (int k = 0; k < num_k; k++) {
                for (int ci_t = 0; ci_t < C_in_tiles; ci_t++) {
                    for (int co_t = 0; co_t < C_out_tiles; co_t++) {
                        uint32_t* block = packed.data() + 
                            ((size_t)k * C_in_tiles * C_out_tiles + ci_t * C_out_tiles + co_t) * 128;
                        
                        // Fill 8 rows × 16 cols
                        for (int row = 0; row < 8; row++) {
                            for (int col = 0; col < 16; col++) {
                                int k_idx_0 = ci_t * 16 + row * 2;      // first K element
                                int k_idx_1 = ci_t * 16 + row * 2 + 1;  // second K element
                                int n_idx = co_t * 8 + (col % 8);       // output channel
                                
                                uint16_t h0 = src[k * C_in * C_out + k_idx_0 * C_out + n_idx];
                                uint16_t h1 = src[k * C_in * C_out + k_idx_1 * C_out + n_idx];
                                
                                block[row * 16 + col] = ((uint32_t)h1 << 16) | (uint32_t)h0;
                            }
                        }
                    }
                }
            }
            
            size_t pb = packed_count * sizeof(uint32_t);
            g_layer_weights_dpas[i] = clCreateBuffer(g_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                      pb, packed.data(), &clerr);
            CL_CHECK(clerr);
        }
        std::cout << "[sparse_encoder_ext] DPAS weight packing done" << std::endl;
    }

    // ── Transposed weight packing ──
    // Note: Transposed weight layouts (w4t, w8t) were tested but proved slower
    // due to destroyed cross-group cache sharing within sub-groups. Removed.
}

// ============================================================
// Morton code utility for spatial sorting of voxels
// ============================================================
static inline uint32_t spread_bits_3(uint32_t v) {
    v &= 0x3FF;  // 10 bits max
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

static inline uint32_t morton3d(int x, int y, int z) {
    return spread_bits_3((uint32_t)x) | (spread_bits_3((uint32_t)y) << 1) | (spread_bits_3((uint32_t)z) << 2);
}

// Sort voxel coordinates by Morton code and return the permutation
// perm[new_idx] = old_idx (so features must be reordered: new_feat[new_idx] = old_feat[perm[new_idx]])
static void sort_voxels_morton(const int* coords, int N, std::vector<int>& perm) {
    // Compute Morton codes
    std::vector<std::pair<uint32_t, int>> coded(N);
    for (int i = 0; i < N; i++) {
        coded[i] = {morton3d(coords[i*4+1], coords[i*4+2], coords[i*4+3]), i};
    }
    std::sort(coded.begin(), coded.end());
    perm.resize(N);
    for (int i = 0; i < N; i++) {
        perm[i] = coded[i].second;
    }
}


// ============================================================
// Main evaluate() — full sparse encoder pipeline
// Pre-computes ALL coordinate maps upfront, then runs GPU
// convolutions without CPU/GPU synchronization barriers.
// ============================================================
bool SparseEncoderOp::evaluate(
    ov::TensorVector& outputs,
    const ov::TensorVector& inputs
) const {
    std::call_once(g_init_flag, init_opencl);

    auto t_start = std::chrono::high_resolution_clock::now();
    double ms_precomp = 0, ms_subm = 0, ms_sgpu = 0, ms_s2d = 0;

    const float* features  = inputs[0].data<float>();
    const int*   coords    = inputs[1].data<int>();
    const int    N         = inputs[2].data<int>()[0];
    const float* params    = inputs[3].data<float>();
    float*       bev_out   = outputs[0].data<float>();

    if (N <= 0) {
        std::memset(bev_out, 0, 1 * SPENC_BEV_C * SPENC_BEV_H * SPENC_BEV_W * sizeof(float));
        return true;
    }

    // Compute param offsets
    int w_offsets[NUM_LAYERS], s_offsets[NUM_LAYERS], b_offsets[NUM_LAYERS];
    {
        int woff = 0, soff = 0;
        int total_w = 0, total_s = 0;
        for (int i = 0; i < NUM_LAYERS; i++) total_w += LAYERS[i].num_k * LAYERS[i].C_in * LAYERS[i].C_out;
        for (int i = 0; i < NUM_LAYERS; i++) total_s += LAYERS[i].C_out;
        int scale_base = total_w;
        int bias_base  = total_w + total_s;
        woff = 0; soff = 0;
        for (int i = 0; i < NUM_LAYERS; i++) {
            w_offsets[i] = woff;
            s_offsets[i] = scale_base + soff;
            b_offsets[i] = bias_base + soff;
            woff += LAYERS[i].num_k * LAYERS[i].C_in * LAYERS[i].C_out;
            soff += LAYERS[i].C_out;
        }
    }

    // Pre-upload weights once (persistent GPU buffers)
    pre_upload_weights(params, w_offsets, s_offsets, b_offsets);

    // ══════════════════════════════════════════════════════════
    // Interleaved CPU precomp + GPU execution
    // Level 0 neighbor map built on GPU (~2ms vs ~50ms CPU)
    // ══════════════════════════════════════════════════════════
    auto t_precomp0 = std::chrono::high_resolution_clock::now();

    // Store level info for later use
    struct LevelInfo {
        const int* coords_ptr;
        std::vector<int> owned_coords;  // for levels > 0
        int N, X, Y, Z;
    };
    LevelInfo levels[4];
    levels[0].coords_ptr = nullptr; // set after Morton sort below
    levels[0].N = N;
    levels[0].X = 1440; levels[0].Y = 1440; levels[0].Z = 41;
    
    StridedConvPrecomp strided_precomp[4];

    auto t_precomp1 = std::chrono::high_resolution_clock::now();
    ms_precomp = std::chrono::duration<double, std::milli>(t_precomp1 - t_precomp0).count();

    // ══════════════════════════════════════════════════════════
    // GPU pipeline (interleaved with CPU precomp of next levels)
    // ══════════════════════════════════════════════════════════
    int cur_N = N;
    int cur_C = SPENC_NUM_FEATURES;

    // ── Morton sort level 0 for cache locality ──
    bool use_morton = false;  // Morton ordering: no measurable GPU benefit, adds CPU overhead
    std::vector<int> sorted_coords_l0;
    std::vector<float> sorted_features_l0;
    if (use_morton) {
        std::vector<int> l0_perm;
        sort_voxels_morton(coords, N, l0_perm);
        sorted_coords_l0.resize(N * 4);
        sorted_features_l0.resize(N * cur_C);
        for (int i = 0; i < N; i++) {
            memcpy(&sorted_coords_l0[i*4], &coords[l0_perm[i]*4], 4*sizeof(int));
            memcpy(&sorted_features_l0[i*cur_C], &features[l0_perm[i]*cur_C], cur_C*sizeof(float));
        }
        levels[0].coords_ptr = sorted_coords_l0.data();
    } else {
        levels[0].coords_ptr = coords;
    }
    const float* upload_features = use_morton ? sorted_features_l0.data() : features;
    const int* upload_coords = use_morton ? sorted_coords_l0.data() : coords;

    // Allocate GPU feature buffers
    size_t max_fp16 = (size_t)N * 128 * sizeof(uint16_t);
    size_t init_f32 = (size_t)N * cur_C * sizeof(float);
    size_t need = std::max(max_fp16, (size_t)N * cur_C * sizeof(uint16_t));
    ensure_buf(g_feat_buf_a, g_fab_sz, need, CL_MEM_READ_WRITE);
    ensure_buf(g_feat_buf_b, g_fbb_sz, need, CL_MEM_READ_WRITE);

    // Upload features FP32 → FP16
    ensure_buf(g_temp_fp32_buf, g_t32_sz, init_f32, CL_MEM_READ_WRITE);
    upload(g_temp_fp32_buf, upload_features, init_f32);
    gpu_f32_to_f16(g_temp_fp32_buf, g_feat_buf_a, N * cur_C);

    // Upload level-0 coords and build neighbor map on GPU
    {
        size_t cb = (size_t)N * 4 * sizeof(int);
        ensure_buf(g_level_coords_buf[0], g_level_coords_sz[0], cb, CL_MEM_READ_WRITE);
        upload(g_level_coords_buf[0], upload_coords, cb);
        size_t nb = (size_t)N * 27 * sizeof(int);
        ensure_buf(g_level_nmap_buf[0], g_level_nmap_sz[0], nb, CL_MEM_READ_WRITE);
        // Build neighbor map on GPU (compact hash table)
        gpu_build_nmap(g_level_coords_buf[0], g_level_nmap_buf[0], N, 1440, 1440, 41);
    }

    // Detailed section timing (enable to profile GPU kernel times)
    bool do_section_timing = false;
    double sec_upload = 0, sec_conv0 = 0, sec_level[4] = {}, sec_strided[4] = {}, sec_bev = 0;
    auto t_sec = std::chrono::high_resolution_clock::now();
    if (do_section_timing) {
        CL_CHECK(clFinish(g_queue));
        sec_upload = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_sec).count();
        t_sec = std::chrono::high_resolution_clock::now();
    }

    // Set active neighbor map/coords to level 0
    g_neighbor_map_buf = g_level_nmap_buf[0];
    g_coords_buf = g_level_coords_buf[0];

    cl_mem cur_feat = g_feat_buf_a;
    cl_mem alt_feat = g_feat_buf_b;

    // ── Layer 0: conv_input 5→16 (SubM with ReLU) ──
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        int Cout = LAYERS[0].C_out;
        run_subm_conv(cur_feat, alt_feat, cur_N, cur_C, Cout, 1, 0);
        std::swap(cur_feat, alt_feat);
        cur_C = Cout;
        if (do_section_timing) {
            CL_CHECK(clFinish(g_queue));
            sec_conv0 = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_sec).count();
            t_sec = std::chrono::high_resolution_clock::now();
        }
        ms_subm += std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }

    // ── Start GPU block 0 processing, then compute rest of coord maps ──
    // Flush to start GPU execution of block 0
    CL_CHECK(clFlush(g_queue));

    // Process layers 1..20 with just-in-time precomputation
    int li = 1;
    int cur_level = 0;      // current resolution level
    int strided_idx = 0;    // which strided conv we're at
    
    // Track which strided levels have been precomputed
    bool strided_computed[4] = {false, false, false, false};
    bool level_nmap_computed[4] = {true, false, false, false}; // level 0 done
    
    while (li < NUM_LAYERS) {
        const auto& L = LAYERS[li];

        if (L.type == 1) {
            // ─── Residual block (2 convs) ───
            auto t0 = std::chrono::high_resolution_clock::now();
            int Cin = L.C_in;
            int Cout = LAYERS[li+1].C_out;

            // Save identity
            size_t id_bytes = (size_t)cur_N * Cin * sizeof(uint16_t);
            ensure_buf(g_identity_buf, g_idb_sz, id_bytes, CL_MEM_READ_WRITE);

            // Conv1: cur_feat(=identity) → alt_feat [with ReLU]
            run_subm_conv(cur_feat, alt_feat, cur_N, Cin, LAYERS[li].C_out, 1, li);

            // Choose: fused conv2+resadd or separate conv2 + resadd
            // Fused kernel: conv2(alt_feat) + identity(cur_feat) → cur_feat
            // Only use fused kernel for channel sizes where lite (4-wide) is optimal
            cl_kernel resadd_kern = nullptr;
            int resadd_div = 4;
            if (g_subm_conv_fp16_lite_resadd_kernel && Cout >= 32) {
                resadd_kern = g_subm_conv_fp16_lite_resadd_kernel;
                resadd_div = 4;
            }

            if (resadd_kern) {
                // Fused conv2+resadd+ReLU: alt_feat → cur_feat (identity=cur_feat)
                dispatch_resadd_kernel(resadd_kern, alt_feat, cur_feat,
                                       cur_feat, cur_N,
                                       LAYERS[li+1].C_in, Cout,
                                       li+1, resadd_div);
            } else {
                // Fallback: separate conv2 + resadd
                run_subm_conv(alt_feat, g_identity_buf, cur_N, LAYERS[li+1].C_in, Cout, 0, li+1);
                int total = cur_N * Cout;
                CL_CHECK(clSetKernelArg(g_residual_add_relu_fp16_kernel, 0, sizeof(cl_mem), &g_identity_buf));
                CL_CHECK(clSetKernelArg(g_residual_add_relu_fp16_kernel, 1, sizeof(cl_mem), &cur_feat));
                CL_CHECK(clSetKernelArg(g_residual_add_relu_fp16_kernel, 2, sizeof(cl_mem), &cur_feat));
                CL_CHECK(clSetKernelArg(g_residual_add_relu_fp16_kernel, 3, sizeof(int), &total));
                size_t gws = (size_t)total;
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_residual_add_relu_fp16_kernel,
                                                 1, nullptr, &gws, nullptr, 0, nullptr, nullptr));
            }

            cur_C = Cout;
            ms_subm += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t0).count();
            li += 2;
            
            // Section timing after residual block
            if (do_section_timing && (li == 5 || li == 10 || li == 15 || li == 20)) {
                CL_CHECK(clFinish(g_queue));
                sec_level[cur_level] += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_sec).count();
                t_sec = std::chrono::high_resolution_clock::now();
            }
            
            // After submitting a residual block, compute next strided level's
            // coordinate maps on CPU (overlaps with GPU kernel execution).
            // We precompute ALL remaining levels here since the GPU is busy
            // with subm conv kernels and won't need the data until later.
            if (strided_idx < 4 && !strided_computed[strided_idx]) {
                auto t_cpu0 = std::chrono::high_resolution_clock::now();
                
                // Compute strided conv output coordinates
                int si = strided_idx;
                const int strided_layers[] = {5, 10, 15, 20};
                int sli = strided_layers[si];
                const auto& SL = LAYERS[sli];
                
                int cur_X = levels[si].X, cur_Y = levels[si].Y, cur_Z = levels[si].Z;
                int X_out = (cur_X + 2*SL.px - SL.kx) / SL.sx + 1;
                int Y_out = (cur_Y + 2*SL.py - SL.ky) / SL.sy + 1;
                int Z_out = (cur_Z + 2*SL.pz - SL.kz) / SL.sz + 1;
                
                int src_N = levels[si].N;
                const int* src_coords = levels[si].coords_ptr;
                
                size_t oh_entries = (size_t)X_out * Y_out * Z_out;
                std::vector<int> h_out_hash(oh_entries, -1);
                std::vector<int> out_coords;
                out_coords.reserve(src_N * 4);
                int next_idx = 0;
                
                for (int i = 0; i < src_N; i++) {
                    int batch = src_coords[i*4+0];
                    int in_x = src_coords[i*4+1], in_y = src_coords[i*4+2], in_z = src_coords[i*4+3];
                    for (int ki = 0; ki < SL.kx; ki++)
                        for (int kj = 0; kj < SL.ky; kj++)
                            for (int kk = 0; kk < SL.kz; kk++) {
                                int ox = in_x + SL.px - ki, oy = in_y + SL.py - kj, oz = in_z + SL.pz - kk;
                                if (ox >= 0 && ox % SL.sx == 0 && oy >= 0 && oy % SL.sy == 0 &&
                                    oz >= 0 && oz % SL.sz == 0) {
                                    int out_x = ox/SL.sx, out_y = oy/SL.sy, out_z = oz/SL.sz;
                                    if (out_x >= 0 && out_x < X_out &&
                                        out_y >= 0 && out_y < Y_out &&
                                        out_z >= 0 && out_z < Z_out) {
                                        int64_t key = ((int64_t)out_x * Y_out + out_y) * Z_out + out_z;
                                        if (h_out_hash[key] == -1) {
                                            h_out_hash[key] = next_idx++;
                                            out_coords.push_back(batch);
                                            out_coords.push_back(out_x);
                                            out_coords.push_back(out_y);
                                            out_coords.push_back(out_z);
                                        }
                                    }
                                }
                            }
                }
                int N_out = (int)out_coords.size() / 4;
                
                // Morton-sort output coordinates for cache locality
                if (use_morton) {
                    std::vector<int> morton_perm;
                    sort_voxels_morton(out_coords.data(), N_out, morton_perm);
                    std::vector<int> sorted_coords(N_out * 4);
                    for (int i = 0; i < N_out; i++) {
                        memcpy(&sorted_coords[i*4], &out_coords[morton_perm[i]*4], 4*sizeof(int));
                    }
                    strided_precomp[si].out_coords = sorted_coords;
                } else {
                    strided_precomp[si].out_coords = out_coords;
                }
                strided_precomp[si].N_out = N_out;
                strided_precomp[si].X_out = X_out;
                strided_precomp[si].Y_out = Y_out;
                strided_precomp[si].Z_out = Z_out;
                strided_computed[si] = true;
                
                // Set up next level info
                if (si < 3) {
                    levels[si+1].owned_coords = strided_precomp[si].out_coords;
                    levels[si+1].coords_ptr = levels[si+1].owned_coords.data();
                    levels[si+1].N = N_out;
                    levels[si+1].X = X_out;
                    levels[si+1].Y = Y_out;
                    levels[si+1].Z = Z_out;
                }
                
                ms_precomp += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_cpu0).count();
            }
        }
        else if (L.type == 2 || L.type == 3) {
            // ─── Strided conv — NMAP GATHER approach ───
            // Precompute neighbor map on GPU, then run vectorized conv
            // with w4 (4 output channels per WI). No hash lookups in the
            // main conv kernel → much faster than direct gather.
            auto t_gpu0 = std::chrono::high_resolution_clock::now();

            int Cin = L.C_in, Cout = L.C_out;
            auto& sp = strided_precomp[strided_idx];

            // The INPUT hash table (from nmap building) is still in g_hash_keys_buf.
            int input_table_mask = g_last_table_mask;
            int X_in = levels[cur_level].X;
            int Y_in = levels[cur_level].Y;
            int Z_in = levels[cur_level].Z;

            // Upload output coords
            size_t out_cb = (size_t)sp.N_out * 4 * sizeof(int);
            cl_mem out_coords_buf;
            if (L.type != 3 && cur_level + 1 < MAX_LEVELS) {
                ensure_buf(g_level_coords_buf[cur_level + 1], g_level_coords_sz[cur_level + 1], out_cb, CL_MEM_READ_WRITE);
                upload(g_level_coords_buf[cur_level + 1], sp.out_coords.data(), out_cb);
                out_coords_buf = g_level_coords_buf[cur_level + 1];
            } else {
                ensure_buf(g_strided_hash_buf, g_shb_sz, out_cb, CL_MEM_READ_WRITE);
                upload(g_strided_hash_buf, sp.out_coords.data(), out_cb);
                out_coords_buf = g_strided_hash_buf;
            }

            // Allocate output FP16 buffer — must differ from input (cur_feat)
            size_t out_fp16b = (size_t)sp.N_out * Cout * sizeof(uint16_t);
            ensure_buf(g_feat_buf_a, g_fab_sz, out_fp16b, CL_MEM_READ_WRITE);
            ensure_buf(g_feat_buf_b, g_fbb_sz, out_fp16b, CL_MEM_READ_WRITE);
            cl_mem gather_out = (cur_feat == g_feat_buf_a) ? g_feat_buf_b : g_feat_buf_a;

            int kx = L.kx, ky = L.ky, kz = L.kz;
            int sx = L.sx, sy = L.sy, sz = L.sz;
            int px = L.px, py = L.py, pz = L.pz;
            int num_k = kx * ky * kz;

            if (g_build_strided_nmap_kernel && g_strided_conv_nmap_fp16_kernel) {
                // ── Build strided neighbor map on GPU (hash lookups done ONCE) ──
                size_t snb = (size_t)sp.N_out * num_k * sizeof(int);
                ensure_buf(g_strided_nmap_buf[strided_idx], g_strided_nmap_sz[strided_idx],
                           snb, CL_MEM_READ_WRITE);

                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 0, sizeof(cl_mem), &out_coords_buf));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 1, sizeof(cl_mem), &g_hash_keys_buf));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 2, sizeof(cl_mem), &g_hash_vals_buf));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 3, sizeof(cl_mem), &g_strided_nmap_buf[strided_idx]));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 4, sizeof(int), &sp.N_out));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 5, sizeof(int), &kx));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 6, sizeof(int), &ky));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 7, sizeof(int), &kz));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 8, sizeof(int), &sx));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 9, sizeof(int), &sy));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 10, sizeof(int), &sz));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 11, sizeof(int), &px));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 12, sizeof(int), &py));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 13, sizeof(int), &pz));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 14, sizeof(int), &X_in));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 15, sizeof(int), &Y_in));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 16, sizeof(int), &Z_in));
                CL_CHECK(clSetKernelArg(g_build_strided_nmap_kernel, 17, sizeof(int), &input_table_mask));
                size_t gws_nmap = (size_t)sp.N_out;
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_build_strided_nmap_kernel,
                                                 1, nullptr, &gws_nmap, nullptr, 0, nullptr, nullptr));

                // ── Vectorized strided conv with precomputed nmap (w4) ──
                int apply_relu = 1;
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 0, sizeof(cl_mem), &cur_feat));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 1, sizeof(cl_mem), &g_strided_nmap_buf[strided_idx]));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 2, sizeof(cl_mem), &g_layer_weights[li]));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 3, sizeof(cl_mem), &g_layer_scales[li]));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 4, sizeof(cl_mem), &g_layer_biases[li]));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 5, sizeof(cl_mem), &gather_out));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 6, sizeof(int), &sp.N_out));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 7, sizeof(int), &Cin));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 8, sizeof(int), &Cout));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 9, sizeof(int), &num_k));
                CL_CHECK(clSetKernelArg(g_strided_conv_nmap_fp16_kernel, 10, sizeof(int), &apply_relu));

                int cout_div4 = (Cout + 3) / 4;
                size_t gws2[2] = {(size_t)cout_div4, (size_t)sp.N_out};
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_strided_conv_nmap_fp16_kernel,
                                                 2, nullptr, gws2, nullptr, 0, nullptr, nullptr));
            } else {
                // Fallback: original gather with hash lookups
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 0, sizeof(cl_mem), &cur_feat));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 1, sizeof(cl_mem), &out_coords_buf));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 2, sizeof(cl_mem), &g_layer_weights[li]));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 3, sizeof(cl_mem), &g_layer_scales[li]));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 4, sizeof(cl_mem), &g_layer_biases[li]));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 5, sizeof(cl_mem), &gather_out));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 6, sizeof(cl_mem), &g_hash_keys_buf));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 7, sizeof(cl_mem), &g_hash_vals_buf));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 8, sizeof(int), &sp.N_out));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 9, sizeof(int), &Cin));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 10, sizeof(int), &Cout));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 11, sizeof(int), &kx));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 12, sizeof(int), &ky));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 13, sizeof(int), &kz));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 14, sizeof(int), &sx));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 15, sizeof(int), &sy));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 16, sizeof(int), &sz));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 17, sizeof(int), &px));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 18, sizeof(int), &py));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 19, sizeof(int), &pz));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 20, sizeof(int), &X_in));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 21, sizeof(int), &Y_in));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 22, sizeof(int), &Z_in));
                CL_CHECK(clSetKernelArg(g_strided_conv_gather_fp16_kernel, 23, sizeof(int), &input_table_mask));
                size_t gws2[2] = {(size_t)sp.N_out, (size_t)Cout};
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_strided_conv_gather_fp16_kernel,
                                                 2, nullptr, gws2, nullptr, 0, nullptr, nullptr));
            }

            cur_feat = gather_out;
            alt_feat = (gather_out == g_feat_buf_a) ? g_feat_buf_b : g_feat_buf_a;
            cur_N = sp.N_out;
            cur_C = Cout;

            // ── Morton permutation: reorder features to match sorted coords ──
            // This dramatically improves cache locality for neighbor lookups
            // because spatially nearby voxels are now adjacent in memory.
            if (g_permute_features_fp16_kernel && L.type != 3 && !sp.morton_perm.empty()) {
                size_t perm_bytes = (size_t)sp.N_out * sizeof(int);
                static cl_mem g_perm_buf = nullptr;
                static size_t g_perm_sz = 0;
                ensure_buf(g_perm_buf, g_perm_sz, perm_bytes, CL_MEM_READ_WRITE);
                upload(g_perm_buf, sp.morton_perm.data(), perm_bytes);

                // Permute: alt_feat[i] = cur_feat[perm[i]] for each channel group
                CL_CHECK(clSetKernelArg(g_permute_features_fp16_kernel, 0, sizeof(cl_mem), &cur_feat));
                CL_CHECK(clSetKernelArg(g_permute_features_fp16_kernel, 1, sizeof(cl_mem), &g_perm_buf));
                CL_CHECK(clSetKernelArg(g_permute_features_fp16_kernel, 2, sizeof(cl_mem), &alt_feat));
                CL_CHECK(clSetKernelArg(g_permute_features_fp16_kernel, 3, sizeof(int), &sp.N_out));
                CL_CHECK(clSetKernelArg(g_permute_features_fp16_kernel, 4, sizeof(int), &Cout));
                int n_groups = sp.N_out * (Cout / 8);
                size_t gws_perm = (size_t)n_groups;
                CL_CHECK(clEnqueueNDRangeKernel(g_queue, g_permute_features_fp16_kernel,
                                                 1, nullptr, &gws_perm, nullptr, 0, nullptr, nullptr));
                // Swap: permuted features are now in alt_feat
                cur_feat = alt_feat;
                alt_feat = gather_out;
            }

            // Switch to next level's neighbor map/coords
            cur_level++;
            if (L.type != 3 && cur_level < MAX_LEVELS) {
                auto& nlevel = levels[cur_level];
                // Build OUTPUT hash from output coords (for next level nmap)
                int out_table_mask = gpu_build_hash_table(out_coords_buf, nlevel.N, nlevel.Y, nlevel.Z);
                // Build nmap from output hash
                size_t nb = (size_t)nlevel.N * 27 * sizeof(int);
                ensure_buf(g_level_nmap_buf[cur_level], g_level_nmap_sz[cur_level], nb, CL_MEM_READ_WRITE);
                gpu_build_nmap_from_hash(g_level_coords_buf[cur_level], g_level_nmap_buf[cur_level],
                                         nlevel.N, nlevel.X, nlevel.Y, nlevel.Z, out_table_mask);
                g_coords_buf = g_level_coords_buf[cur_level];
                g_neighbor_map_buf = g_level_nmap_buf[cur_level];
            }

            ms_sgpu += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_gpu0).count();

            // Section timing after strided conv
            if (do_section_timing) {
                CL_CHECK(clFinish(g_queue));
                sec_strided[strided_idx] = std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - t_sec).count();
                t_sec = std::chrono::high_resolution_clock::now();
            }

            strided_idx++;
            li++;
        }
        else {
            li++;
        }
    }

    // ── Sparse to BEV: [N, C] (FP16) → [1, C*Z, X, Y] (FP32) ──
    // Optimized: single memset + vectorized scatter
    auto t_s2d0 = std::chrono::high_resolution_clock::now();

    int fin_X = strided_precomp[3].X_out;
    int fin_Y = strided_precomp[3].Y_out;
    int fin_Z = strided_precomp[3].Z_out;
    size_t bev_elems = (size_t)1 * cur_C * fin_Z * fin_X * fin_Y;
    size_t bev_bytes = bev_elems * sizeof(float);
    size_t bev_fp16_bytes = bev_elems * sizeof(uint16_t);
    int XY = fin_X * fin_Y;

    {
        // ── Optimized CPU scatter: FP16 features → dense FP32 BEV ──
        // The FP16 feature readback is only ~1.8MB (tiny vs 33MB dense BEV).
        // OpenMP parallelizes the scatter + memset across voxels.

        // Read sparse features from GPU: small transfer (~1.8MB FP16)
        // This also drains the GPU pipeline (blocking CL_TRUE).
        size_t feat_bytes = (size_t)cur_N * cur_C * sizeof(uint16_t);
        static std::vector<uint16_t> s2d_feat_buf;
        if (s2d_feat_buf.size() < (size_t)cur_N * cur_C)
            s2d_feat_buf.resize((size_t)cur_N * cur_C);

        CL_CHECK(clEnqueueReadBuffer(g_queue, cur_feat, CL_TRUE, 0, feat_bytes,
                                      s2d_feat_buf.data(), 0, nullptr, nullptr));

        const int* coords = strided_precomp[3].out_coords.data();

        // Zero the output BEV (parallel across threads for bandwidth)
        std::memset(bev_out, 0, bev_bytes);

        // Scatter with OpenMP: each voxel writes to independent BEV positions
        // Thread count controlled by OMP_NUM_THREADS env var (4 threads optimal on 16-core PTL)
        #pragma omp parallel for schedule(static)
        for (int idx = 0; idx < cur_N; idx++) {
            int x = coords[idx * 4 + 1];
            int y = coords[idx * 4 + 2];
            int z = coords[idx * 4 + 3];
            if (x < 0 || x >= fin_X || y < 0 || y >= fin_Y || z < 0 || z >= fin_Z) continue;

            const uint16_t* src = &s2d_feat_buf[idx * cur_C];
            int base_offset = z * XY + x * fin_Y + y;

            for (int c = 0; c < cur_C; c++) {
                uint16_t h = src[c];
                if (h == 0) continue;
                uint32_t exp_bits = (h >> 10) & 0x1F;
                if (exp_bits == 0) continue;
                uint32_t sign = ((uint32_t)h & 0x8000) << 16;
                uint32_t mant = h & 0x3FF;
                uint32_t f = sign | ((exp_bits + 112) << 23) | (mant << 13);
                float val;
                std::memcpy(&val, &f, 4);
                bev_out[(c * fin_Z) * XY + base_offset] = val;
            }
        }
    }

    auto t_s2d1 = std::chrono::high_resolution_clock::now();
    ms_s2d = std::chrono::duration<double, std::milli>(t_s2d1 - t_s2d0).count();
    if (do_section_timing) {
        sec_bev = ms_s2d;
    }

    double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_start).count();

    g_call_count++;
    if (g_call_count <= 5 || g_call_count % 20 == 0) {
        std::cout << "[sparse_encoder_ext] " << elapsed << " ms, N=" << N
                  << " -> N_out=" << cur_N
                  << " | precomp=" << ms_precomp << " subm=" << ms_subm
                  << " sgpu=" << ms_sgpu
                  << " s2d=" << ms_s2d << std::endl;
        if (do_section_timing) {
            std::cout << "[SECTION_PROFILE] upload=" << sec_upload
                      << " conv0=" << sec_conv0
                      << " lv0=" << sec_level[0]
                      << " st0=" << sec_strided[0]
                      << " lv1=" << sec_level[1]
                      << " st1=" << sec_strided[1]
                      << " lv2=" << sec_level[2]
                      << " st2=" << sec_strided[2]
                      << " lv3=" << sec_level[3]
                      << " st3=" << sec_strided[3]
                      << " bev=" << sec_bev << std::endl;
        }
    }

    return true;
}

}  // namespace BEVFusionExtension
