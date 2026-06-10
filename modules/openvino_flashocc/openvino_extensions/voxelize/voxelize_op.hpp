// Voxelization Operations for OpenVINO
//
// Three-layer GPU-accelerated voxelization:
//   1. VoxelizeZero:    zero-initialize workspace buffer
//   2. VoxelizeScatter: hash-table insert + int32 atomic feature accumulation
//   3. VoxelizeMean:    int32 fixed-point → float32 mean features + coords
//
// Replaces the torch/OpenCL voxel_layer C++ extension.

#pragma once

#include <openvino/op/op.hpp>

namespace BEVFusionExtension {

// ── Constants (BEVFusion defaults) ───────────────────────────────────────────
constexpr int64_t VOXEL_MAX_POINTS   = 70000;   // max input point count
constexpr int64_t VOXEL_MAX_VOXELS   = 60000;   // max output voxel count
constexpr int64_t VOXEL_MAX_PTS_PER  = 10;      // max points per voxel
constexpr int64_t VOXEL_NUM_FEATURES = 5;       // x, y, z, intensity, timestamp
constexpr int64_t VOXEL_HASH_SIZE    = 131072;  // 2^17, open-address hash table
constexpr int64_t VOXEL_FP_SCALE     = 4096;    // fixed-point scale

// Hash table entry stride (inline: key, feat*5, count, batch, x, y, z)
constexpr int64_t VOXEL_ENTRY_STRIDE     = 11;
constexpr int64_t VOXEL_HT_OFFSET        = 1;
constexpr int64_t VOXEL_WORKSPACE_SIZE   = 1 + VOXEL_HASH_SIZE * VOXEL_ENTRY_STRIDE; // 1441793

// Grid dimensions
constexpr int VOXEL_GRID_X = 1440;
constexpr int VOXEL_GRID_Y = 1440;
constexpr int VOXEL_GRID_Z = 41;


/**
 * VoxelizeZero Operation
 *
 * Zeros the workspace buffer to prepare for scatter.
 * Needed because OV reuses GPU buffers across inferences.
 *
 * Inputs:
 *   0: trigger      [1]                  i32 — dummy input to establish data flow
 *
 * Outputs:
 *   0: workspace   [WORKSPACE_SIZE]       i32 — all zeros
 */
class VoxelizeZero : public ov::op::Op {
public:
    OPENVINO_OP("VoxelizeZero", "bevfusion");

    VoxelizeZero() = default;
    VoxelizeZero(const ov::Output<ov::Node>& trigger);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};


/**
 * VoxelizeScatter Operation
 *
 * GPU hash-table insert + int32 atomic feature accumulation.
 *
 * Inputs:
 *   0: points      [MAX_POINTS, 5]  f32 — padded LiDAR points
 *   1: num_points   [1]             i32 — actual point count
 *
 * Outputs:
 *   0: workspace   [WORKSPACE_SIZE]  i32 — packed hash table + voxel data
 */
class VoxelizeScatter : public ov::op::Op {
public:
    OPENVINO_OP("VoxelizeScatter", "bevfusion");

    VoxelizeScatter() = default;

    VoxelizeScatter(const ov::Output<ov::Node>& points,
                    const ov::Output<ov::Node>& num_points);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};


/**
 * VoxelizeMean Operation
 *
 * Converts int32 fixed-point accumulators to float32 mean features.
 * Row 0 of output contains num_voxels metadata.
 *
 * Inputs:
 *   0: workspace   [WORKSPACE_SIZE]       i32
 *
 * Outputs:
 *   0: result      [MAX_VOXELS + 1, 9]    f32
 *      Row 0:   [num_voxels, 0, ..., 0]
 *      Row 1+:  [feat0..feat4, batch, coord_x, coord_y, coord_z]
 */
class VoxelizeMean : public ov::op::Op {
public:
    OPENVINO_OP("VoxelizeMean", "bevfusion");

    VoxelizeMean() = default;

    VoxelizeMean(const ov::Output<ov::Node>& workspace);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace BEVFusionExtension
