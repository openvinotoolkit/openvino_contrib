// CdpnCoordDenorm GPU kernel - denormalise xyz coordinates by obj_extents
// and min-max normalise the confidence channel.
//
// Tensors (4D BFYX):
//   input0  coord_maps:  [1, 4, R, R]   f32  (ch0-2: norm coords, ch3: raw conf)
//   input1  obj_extents: [1, 1, 1, 3]   f32  (abs_x, abs_y, abs_z)
//   output0 result:      [1, 4, R, R]   f32  (ch0-2: denorm coords, ch3: norm conf)
//
// Single work-item kernel (global=1). R*R is small (64*64=4096),
// so a serial loop is fast enough and avoids cross-work-group sync issues
// needed for the min-max confidence normalisation.
//
// JIT defines from XML:
//   SPATIAL_R : int (64)

#ifndef SPATIAL_R
#define SPATIAL_R 64
#endif

__kernel void cdpn_coord_denorm_gpu(
    const __global float* restrict coord_maps,
    const __global float* restrict obj_extents,
    __global float* restrict result)
{
    const int R2 = SPATIAL_R * SPATIAL_R;

    const float abs_x = obj_extents[0];
    const float abs_y = obj_extents[1];
    const float abs_z = obj_extents[2];

    // Denormalise xyz channels
    for (int i = 0; i < R2; ++i) {
        result[0 * R2 + i] = coord_maps[0 * R2 + i] * abs_x;
        result[1 * R2 + i] = coord_maps[1 * R2 + i] * abs_y;
        result[2 * R2 + i] = coord_maps[2 * R2 + i] * abs_z;
    }

    // Min-max normalise confidence channel
    const __global float* raw_conf = coord_maps + 3 * R2;
    __global float* conf_out = result + 3 * R2;

    float cmin = raw_conf[0];
    float cmax = cmin;
    for (int i = 1; i < R2; ++i) {
        float v = raw_conf[i];
        cmin = min(cmin, v);
        cmax = max(cmax, v);
    }
    float range = cmax - cmin;
    if (range < 1e-8f) range = 1.0f;
    float inv_range = 1.0f / range;

    for (int i = 0; i < R2; ++i) {
        conf_out[i] = (raw_conf[i] - cmin) * inv_range;
    }
}
