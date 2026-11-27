// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __PP_SCATTER_HPP__
#define __PP_SCATTER_HPP__

#include <openvino/op/op.hpp>

#include "pp_voxelization.hpp"

namespace PpExtension {

// One BEV channel plane, i.e. the number of grid cells.
constexpr auto PP_BEV_CELL_COUNT = PP_GRID_Y * PP_GRID_X;

// Scatters compact pillar features into the dense BEV feature map.
// Inputs:
//   pillar_features    [P, C] pillar features from the PFN
//   pillar_coordinates [P, PP_COORD_VALUES] (batch, z, y, x)
//   pillar_count       [1]    pillar count
// Output:
//   bev [1, C, PP_GRID_Y, PP_GRID_X], same element type as pillar_features
class PPScatterBEV : public ov::op::Op {
public:
    OPENVINO_OP("PPScatterBEV");

    PPScatterBEV() = default;
    PPScatterBEV(const ov::Output<ov::Node>& pillar_features,
                 const ov::Output<ov::Node>& pillar_coordinates,
                 const ov::Output<ov::Node>& pillar_count);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace PpExtension

#endif  // __PP_SCATTER_HPP__
