/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

// BEVPool Operations for OpenVINO
// Implements BEV pooling using int32 fixed-point atomics for speed.
//
// Three-layer approach:
//   1. Built-in Softmax (opset8) for depth softmax
//   2. BEVPoolScatter: int32 atomic scatter with pre-computed softmax
//   3. BEVPoolConvert: int32 fixed-point → float32
//
// Also retains original fused BEVPool op for backward compatibility.

#pragma once

#include <openvino/op/op.hpp>

namespace BEVFusionExtension {

/**
 * BEVPool Operation (fused softmax + scatter, float output)
 * Kept for backward compatibility. Uses float CAS atomics.
 */
class BEVPool : public ov::op::Op {
public:
    OPENVINO_OP("BEVPool", "bevfusion");

    BEVPool() = default;

    BEVPool(const ov::Output<ov::Node>& depth_logits,
            const ov::Output<ov::Node>& context_feats,
            const ov::Output<ov::Node>& geom,
            int64_t nx = 360, int64_t ny = 360, int64_t nz = 1,
            int64_t num_cams = 6, int64_t depth_bins = 118,
            int64_t channels = 80, int64_t feat_h = 32, int64_t feat_w = 88,
            float x_min = -54.0f, float y_min = -54.0f, float z_min = -10.0f,
            float x_step = 0.3f, float y_step = 0.3f, float z_step = 20.0f);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_nx{360}, m_ny{360}, m_nz{1};
    int64_t m_num_cams{6}, m_depth_bins{118}, m_channels{80};
    int64_t m_feat_h{32}, m_feat_w{88};
    float m_x_min{-54.0f}, m_y_min{-54.0f}, m_z_min{-10.0f};
    float m_x_step{0.3f}, m_y_step{0.3f}, m_z_step{20.0f};
};


/**
 * BEVPoolScatter Operation
 *
 * Takes pre-computed depth softmax weights and scatters weighted context
 * features into a BEV grid using int32 fixed-point atomic_add.
 *
 * Inputs:
 *   0: depth_probs   - [N, D, H, W]  softmax probabilities (FP32)
 *   1: context_feats - [N, C, H, W]  context features      (FP32)
 *   2: geom          - [N*D*H*W, 3]  world-space coords    (FP32)
 *   3: bev_accum     - [1, C*NZ, NX, NY] int32 accumulator (optional)
 *
 * Outputs:
 *   with input 3: a [1] int32 token that orders BEVPoolConvert after the
 *                 scatter. The scatter adds into input 3 in place, which
 *                 must hold zeros; BEVPoolConvert resets it after reading.
 *   without:      0: bev_accum - [1, C*NZ, NX, NY] int32 accumulator. The
 *                 GPU kernel only adds into it and never zeroes it, so this
 *                 form is only correct on memory that happens to be zero.
 *                 The GPU plugin can hand the op a reused pool buffer, so
 *                 prefer the 4-input form.
 */
class BEVPoolScatter : public ov::op::Op {
public:
    OPENVINO_OP("BEVPoolScatter", "bevfusion");

    BEVPoolScatter() = default;

    BEVPoolScatter(const ov::OutputVector& inputs,
                   int64_t nx = 360, int64_t ny = 360, int64_t nz = 1,
                   int64_t num_cams = 6, int64_t depth_bins = 118,
                   int64_t channels = 80, int64_t feat_h = 32, int64_t feat_w = 88,
                   float x_min = -54.0f, float y_min = -54.0f, float z_min = -10.0f,
                   float x_step = 0.3f, float y_step = 0.3f, float z_step = 20.0f);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_nx{360}, m_ny{360}, m_nz{1};
    int64_t m_num_cams{6}, m_depth_bins{118}, m_channels{80};
    int64_t m_feat_h{32}, m_feat_w{88};
    float m_x_min{-54.0f}, m_y_min{-54.0f}, m_z_min{-10.0f};
    float m_x_step{0.3f}, m_y_step{0.3f}, m_z_step{20.0f};
};


/**
 * BEVPoolConvert Operation
 *
 * Converts int32 fixed-point accumulator to float32.
 * Simple element-wise: output[i] = (float)input[i] / SCALE
 *
 * Inputs, one of:
 *   0: bev_accum - [1, C, NX, NY] int32 fixed-point
 *   0: token     - [1] int32 from the 4-input BEVPoolScatter (orders this op after it)
 *   1: bev_accum - [1, C, NX, NY] int32 fixed-point; reset to zero once read
 *
 * Outputs:
 *   0: bev_float - [1, C, NX, NY] float32
 */
class BEVPoolConvert : public ov::op::Op {
public:
    OPENVINO_OP("BEVPoolConvert", "bevfusion");

    BEVPoolConvert() = default;

    BEVPoolConvert(const ov::OutputVector& inputs,
                   int64_t scale = 8192);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_scale{8192};
};

}  // namespace BEVFusionExtension
