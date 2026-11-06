// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

// Shared constants and helpers for voxelization.

// Scene geometry and metadata offsets come from the generated pillar
// configuration concatenated before this file.
// The metadata buffer stores the pillar counter, compact pillar records, and
// the cell-to-pillar lookup used by the scatter stage.

// Compute grid coordinates for a point. Return false when it is outside the scene.
inline bool point_grid(float x, float y, float z, int* ix, int* iy, int* iz) {
    if (x < MIN_X || x >= MAX_X || y < MIN_Y || y >= MAX_Y || z < MIN_Z || z >= MAX_Z) return false;

    *ix = (int)floor((x - MIN_X) / VOXEL_X);
    *iy = (int)floor((y - MIN_Y) / VOXEL_Y);
    *iz = (int)floor((z - MIN_Z) / VOXEL_Z);
    return *ix >= 0 && *ix < GRID_X && *iy >= 0 && *iy < GRID_Y && *iz >= 0 && *iz < GRID_Z;
}

// Flatten the horizontal grid coordinates into a cell index.
inline int cell_index(int ix, int iy) { return iy * GRID_X + ix; }
