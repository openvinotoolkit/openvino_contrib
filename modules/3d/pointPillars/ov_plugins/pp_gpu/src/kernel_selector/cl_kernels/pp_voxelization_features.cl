// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Voxelization stage 2: generate pillar features.
// ---------------------------------------------------------------------------
// One work-item processes each pillar. Features contain raw values, offsets
// from the pillar mean, and offsets from the voxel center.

inline void write_feature(__global OUTPUT0_TYPE* features, int pillar, int point, int channel, float value) {
    const int index = (pillar * POINTS_PER_VOXEL + point) * OUT_FEATURES + channel;
    features[index] = convert_half_rte(value);
}

inline float voxel_center(float voxel_size, int grid, float min_range) {
    volatile float product = (float)grid * voxel_size;
    volatile float half_step = voxel_size / 2.0f;
    volatile float center = product + half_step;
    center = center + min_range;
    return center;
}

__kernel void pp_voxelization_finalize(__global const INPUT0_TYPE* meta_storage,
                                       __global const INPUT1_TYPE* points,
                                       __global OUTPUT0_TYPE* features) {
    __global const int* meta = (__global const int*)meta_storage;

    const int pillar = get_global_id(0);
    int num_pillars = meta[META_COUNTER];
    if (num_pillars > MAX_VOXELS) num_pillars = MAX_VOXELS;
    if (pillar >= num_pillars) return;

    int count = meta[META_COUNT + pillar];
    if (count <= 0) return;
    count = min(count, POINTS_PER_VOXEL);

    // meta stores the source point index per slot; gather raw features from the
    // points input (kept off the dense buffer to shrink it).
    const int idx0 = META_RAW + pillar * META_RAW_STRIDE;
    float mean_x = 0.0f;
    float mean_y = 0.0f;
    float mean_z = 0.0f;
    for (int point = 0; point < count; point++) {
        const int base = meta[idx0 + point] * POINT_FEATURES;
        mean_x += (float)points[base + 0];
        mean_y += (float)points[base + 1];
        mean_z += (float)points[base + 2];
    }
    mean_x /= (float)count;
    mean_y /= (float)count;
    mean_z /= (float)count;

    const int grid_z = meta[META_COORD + pillar * COORD_VALUES + COORD_Z];
    const int grid_y = meta[META_COORD + pillar * COORD_VALUES + COORD_Y];
    const int grid_x = meta[META_COORD + pillar * COORD_VALUES + COORD_X];
    const float center_x = voxel_center(VOXEL_X, grid_x, MIN_X);
    const float center_y = voxel_center(VOXEL_Y, grid_y, MIN_Y);
    const float center_z = voxel_center(VOXEL_Z, grid_z, MIN_Z);

    for (int point = 0; point < POINTS_PER_VOXEL; point++) {
        if (point >= count) {
            for (int channel = 0; channel < OUT_FEATURES; channel++) write_feature(features, pillar, point, channel, 0.0f);
            continue;
        }
        const int base = meta[idx0 + point] * POINT_FEATURES;
        const float x = (float)points[base + 0];
        const float y = (float)points[base + 1];
        const float z = (float)points[base + 2];
        const float intensity = (float)points[base + 3];

        write_feature(features, pillar, point, 0, x);
        write_feature(features, pillar, point, 1, y);
        write_feature(features, pillar, point, 2, z);
        write_feature(features, pillar, point, 3, intensity);
        write_feature(features, pillar, point, 4, x - mean_x);
        write_feature(features, pillar, point, 5, y - mean_y);
        write_feature(features, pillar, point, 6, z - mean_z);
        write_feature(features, pillar, point, 7, x - center_x);
        write_feature(features, pillar, point, 8, y - center_y);
        write_feature(features, pillar, point, 9, z - center_z);
    }
}
