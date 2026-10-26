/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

// FlexiCubes mesh extraction declaration.
#pragma once

#include <openvino/op/op.hpp>
#include <vector>
#include <cstdint>

namespace SAM3DExtension {

// Standalone extraction (CPU). coords_zyx: N×3 int (z,y,x); feats: N×101 float.
void flexicubes_extract(const int32_t* coords_zyx, const float* feats, int N, int res,
                        std::vector<float>& out_vertices,
                        std::vector<int32_t>& out_faces,
                        std::vector<float>& out_colors);

// FlexiCubes / SparseFeatures2Mesh custom op.
// Inputs:  0: features [MAX_N,101] f32, 1: coords [MAX_N,4] i32, 2: num_voxels [1] i32
// Outputs: 0: vertices [V,3] f32, 1: faces [F,3] i32, 2: colors [V,6] f32, 3: counts [2] i32
class FlexiCubesOp : public ov::op::Op {
public:
    OPENVINO_OP("FlexiCubesExtract", "sam3d");

    FlexiCubesOp() = default;
    FlexiCubesOp(const ov::Output<ov::Node>& features,
                 const ov::Output<ov::Node>& coords,
                 const ov::Output<ov::Node>& num_voxels);
    explicit FlexiCubesOp(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_res = 256;
};

}  // namespace SAM3DExtension
