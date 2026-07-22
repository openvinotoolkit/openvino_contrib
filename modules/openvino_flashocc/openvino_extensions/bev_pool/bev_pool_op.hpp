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
 *
 * Outputs:
 *   0: bev_accum     - [1, C*NZ, NX, NY] int32 fixed-point accumulator
 */
class BEVPoolScatter : public ov::op::Op {
public:
    OPENVINO_OP("BEVPoolScatter", "bevfusion");

    BEVPoolScatter() = default;

    BEVPoolScatter(const ov::Output<ov::Node>& depth_probs,
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
 * BEVPoolConvert Operation
 *
 * Converts int32 fixed-point accumulator to float32.
 * Simple element-wise: output[i] = (float)input[i] / SCALE
 *
 * Inputs:
 *   0: bev_accum - [1, C, NX, NY] int32 fixed-point
 *
 * Outputs:
 *   0: bev_float - [1, C, NX, NY] float32
 */
class BEVPoolConvert : public ov::op::Op {
public:
    OPENVINO_OP("BEVPoolConvert", "bevfusion");

    BEVPoolConvert() = default;

    BEVPoolConvert(const ov::Output<ov::Node>& bev_accum,
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


/**
 * BEVPoolBinSort Operation (GPU counting sort of geometry → packed buffer)
 *
 * Single-workgroup GPU kernel that performs a counting sort on geometry
 * coordinates to produce a single packed output buffer containing:
 *   [sorted_ranks | cell_scratch | interval_starts | interval_lengths]
 *   offsets: 0      TOTAL_PTS     2*TOTAL_PTS      2*TOTAL_PTS+NX*NY
 *
 * Packed into one buffer because SimpleGPU custom layers only support
 * single-output operations.
 *
 * Inputs:
 *   0: geom             - [TOTAL_PTS, 3]   FP32  (world coordinates)
 *
 * Outputs:
 *   0: packed           - [TOTAL_PTS*2 + NX*NY*2]  I32
 */
class BEVPoolBinSort : public ov::op::Op {
public:
    OPENVINO_OP("BEVPoolBinSort", "bevfusion");

    BEVPoolBinSort() = default;

    BEVPoolBinSort(const ov::Output<ov::Node>& geom,
                   int64_t nx = 360, int64_t ny = 360,
                   int64_t total_pts = 1993728,
                   float x_min = -54.0f, float y_min = -54.0f,
                   float x_step = 0.3f, float y_step = 0.3f);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_nx{360}, m_ny{360};
    int64_t m_total_pts{1993728};
    float m_x_min{-54.0f}, m_y_min{-54.0f};
    float m_x_step{0.3f}, m_y_step{0.3f};
};


/**
 * BEVPoolV2 Operation (pre-sorted interval-based scatter, no atomics)
 *
 * Takes a packed sort buffer (from BEVPoolBinSort) and accumulates
 * weighted context features per BEV cell. Context feature indices are
 * computed on-the-fly from the point index.
 *
 * Inputs:
 *   0: depth_probs      - [N, D, H, W]     FP32  (softmax probabilities)
 *   1: context_feats    - [N, C, H, W]     FP32  (context features)
 *   2: packed_sort      - [TOTAL_PTS*2 + NX*NY*2]  I32 (from BEVPoolBinSort)
 *
 * Outputs:
 *   0: bev              - [1, C, NY, NX]   FP32
 */
class BEVPoolV2 : public ov::op::Op {
public:
    OPENVINO_OP("BEVPoolV2", "bevfusion");

    BEVPoolV2() = default;

    BEVPoolV2(const ov::Output<ov::Node>& depth_probs,
              const ov::Output<ov::Node>& context_feats,
              const ov::Output<ov::Node>& packed_sort,
              int64_t nx = 360, int64_t ny = 360,
              int64_t channels = 80, int64_t feat_hw = 2816,
              int64_t depth_hw = 332288,
              int64_t total_pts = 1993728);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;

private:
    int64_t m_nx{360}, m_ny{360};
    int64_t m_channels{80};
    int64_t m_feat_hw{2816};
    int64_t m_depth_hw{332288};
    int64_t m_total_pts{1993728};
};

}  // namespace BEVFusionExtension
