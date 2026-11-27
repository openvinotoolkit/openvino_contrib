// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __PP_VOXELIZATION_HPP__
#define __PP_VOXELIZATION_HPP__

#include <openvino/op/op.hpp>

#include "pp_pillar_config_generated.hpp"

namespace PpExtension {

// Stage 1: fused scatter + pillar compaction.
// Buckets points into grid cells and writes pillar indices, coordinates, and counts.
// Inputs:
//   points [PP_MAX_POINTS, PP_NUM_POINT_FEATURES] f32
//   point_count [1] i32
// Output: metadata [1, 1, 1, PP_META_SIZE] f32.
class PPVoxelizationScatter : public ov::op::Op {
public:
    OPENVINO_OP("PPVoxelizationScatter");

    PPVoxelizationScatter() = default;
    PPVoxelizationScatter(const ov::Output<ov::Node>& points,
                          const ov::Output<ov::Node>& point_count);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 2: finalize pillar features.
// Expands raw pillar points into PP_NUM_OUTPUT_FEATURES f16 features.
// Inputs:
//   metadata [1, 1, 1, PP_META_SIZE] f32
//   points [PP_MAX_POINTS, PP_NUM_POINT_FEATURES] f32
// Output: features [PP_MAX_VOXELS, PP_MAX_POINTS_PER_VOXEL, PP_NUM_OUTPUT_FEATURES] f16.
class PPVoxelizationFinalize : public ov::op::Op {
public:
    OPENVINO_OP("PPVoxelizationFinalize");

    PPVoxelizationFinalize() = default;
    PPVoxelizationFinalize(const ov::Output<ov::Node>& metadata,
                           const ov::Output<ov::Node>& points);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 3: pillar coordinates.
// Input: metadata [1, 1, 1, PP_META_SIZE] f32.
// Output: coordinates [PP_MAX_VOXELS, PP_COORD_VALUES] f32.
class PPVoxelizationCoords : public ov::op::Op {
public:
    OPENVINO_OP("PPVoxelizationCoords");

    PPVoxelizationCoords() = default;
    explicit PPVoxelizationCoords(const ov::Output<ov::Node>& metadata);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 4: pillar count.
// Input: metadata [1, 1, 1, PP_META_SIZE] f32.
// Output: pillar count [1] f32.
class PPVoxelizationParams : public ov::op::Op {
public:
    OPENVINO_OP("PPVoxelizationParams");

    PPVoxelizationParams() = default;
    explicit PPVoxelizationParams(const ov::Output<ov::Node>& metadata);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

// Stage 5: cell-to-pillar lookup.
// Input: metadata [1, 1, 1, PP_META_SIZE] f32.
// Output: lookup table [PP_GRID_SIZE] i32.
// Empty cells contain zero; occupied cells contain the pillar index plus one.
class PPVoxelizationCellPillar : public ov::op::Op {
public:
    OPENVINO_OP("PPVoxelizationCellPillar");

    PPVoxelizationCellPillar() = default;
    explicit PPVoxelizationCellPillar(const ov::Output<ov::Node>& metadata);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace PpExtension

#endif  // __PP_VOXELIZATION_HPP__
