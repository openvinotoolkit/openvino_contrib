/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// Vox2Seq Operation for OpenVINO — SAM 3D Objects
//
// Space-filling curve encoder: maps 3D integer coordinates to 1D codes
// for sort-based serialized attention ordering.
//
// Supports:
//   - Z-order (Morton) encoding: bit-interleave x, y, z
//   - Hilbert curve encoding: LUT-based state machine
//
// Inputs:
//   0: coords   [MAX_N, 3]   I32 — integer 3D coordinates
//   1: num_pts  [1]          I32 — actual point count
//
// Output:
//   0: codes    [MAX_N]      I64 — 1D codes for sorting

#pragma once

#include <openvino/op/op.hpp>
#include <string>

namespace SAM3DExtension {

constexpr int64_t VOX2SEQ_MAX_N = 65536;

class Vox2SeqOp : public ov::op::Op {
public:
    OPENVINO_OP("Vox2Seq", "sam3d");

    Vox2SeqOp() = default;

    Vox2SeqOp(const ov::Output<ov::Node>& coords,
              const ov::Output<ov::Node>& num_pts);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    std::string m_mode = "z_order";  // "z_order" or "hilbert"
    // Axis permutation [px, py, pz] — applied before encoding
    std::vector<int64_t> m_permute = {0, 1, 2};
};

}  // namespace SAM3DExtension
