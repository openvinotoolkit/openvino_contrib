/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
// Point Cloud Operations for OpenVINO
// Custom operations for HGGD grasp detection pipeline
//
// Operations:
//   - KNNPoints: K-nearest neighbors search
//   - BallQuery: Radius-based neighbor search
//   - FPS: Farthest point sampling
//   - MaskedGather: Index-based feature gathering with -1 handling
//   - GatherMaxPool: Fused gather + max pooling

#pragma once

#include <openvino/op/op.hpp>

namespace HGGDExtension {

/**
 * KNNPoints Operation
 *
 * Finds K nearest neighbors for each query point.
 *
 * Inputs:
 *   0: p1 - [B, N1, 3] query points
 *   1: p2 - [B, N2, 3] source points
 *
 * Outputs:
 *   0: dists - [B, N1, K] squared distances
 *   1: idx   - [B, N1, K] indices
 */
class KNNPoints : public ov::op::Op {
public:
    OPENVINO_OP("KNNPoints", "hggd");

    KNNPoints() = default;

    KNNPoints(const ov::Output<ov::Node>& p1,
              const ov::Output<ov::Node>& p2,
              int64_t k = 12);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }

private:
    int64_t m_k{12};
};


/**
 * BallQuery Operation
 *
 * Finds neighbors within radius for each query point.
 * Returns -1 for empty slots.
 *
 * Inputs:
 *   0: p1 - [B, N1, 3] query points
 *   1: p2 - [B, N2, 3] source points
 *
 * Outputs:
 *   0: dists - [B, N1, K] squared distances (-1 for invalid)
 *   1: idx   - [B, N1, K] indices (-1 for invalid)
 */
class BallQuery : public ov::op::Op {
public:
    OPENVINO_OP("BallQuery", "hggd");

    BallQuery() = default;

    BallQuery(const ov::Output<ov::Node>& p1,
              const ov::Output<ov::Node>& p2,
              int64_t k = 32,
              float radius = 0.2f);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }
    float get_radius() const { return m_radius; }

private:
    int64_t m_k{32};
    float m_radius{0.2f};
};


/**
 * FPS (Farthest Point Sampling) Operation
 *
 * Iteratively samples K points that are maximally spread.
 *
 * Inputs:
 *   0: points - [B, N, 3] input points
 *
 * Outputs:
 *   0: sampled - [B, K, 3] sampled points
 *   1: idx     - [B, K] sampled indices
 */
class FPS : public ov::op::Op {
public:
    OPENVINO_OP("FPS", "hggd");

    FPS() = default;

    FPS(const ov::Output<ov::Node>& points,
        int64_t k = 1024);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }

private:
    int64_t m_k{1024};
};


/**
 * MaskedGather Operation
 *
 * Gathers features by indices, handling -1 as invalid.
 *
 * Inputs:
 *   0: points - [B, N, D] source features
 *   1: idx    - [B, M, K] or [B, K] indices (-1 for invalid)
 *
 * Outputs:
 *   0: gathered - [B, M, K, D] or [B, K, D] gathered features
 */
class MaskedGather : public ov::op::Op {
public:
    OPENVINO_OP("MaskedGather", "hggd");

    MaskedGather() = default;

    MaskedGather(const ov::Output<ov::Node>& points,
                 const ov::Output<ov::Node>& idx);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }
};


/**
 * GatherMaxPool Operation
 *
 * Fused gather + max pooling.
 *
 * Inputs:
 *   0: points - [B, N, D] source features
 *   1: idx    - [B, M, K] indices (-1 for invalid)
 *
 * Outputs:
 *   0: output - [B, M, D] max-pooled features
 */
class GatherMaxPool : public ov::op::Op {
public:
    OPENVINO_OP("GatherMaxPool", "hggd");

    GatherMaxPool() = default;

    GatherMaxPool(const ov::Output<ov::Node>& points,
                  const ov::Output<ov::Node>& idx);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }
};


/**
 * PointGather Operation
 *
 * Simple gather for [B, K] indices (e.g., after FPS).
 *
 * Inputs:
 *   0: points - [B, N, D] source features
 *   1: idx    - [B, K] indices
 *
 * Outputs:
 *   0: gathered - [B, K, D] gathered features
 */
class PointGather : public ov::op::Op {
public:
    OPENVINO_OP("PointGather", "hggd");

    PointGather() = default;

    PointGather(const ov::Output<ov::Node>& points,
                const ov::Output<ov::Node>& idx);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }
};


// ═══════════════════════════════════════════════════════════════════════════
// Single-Output Operations for OpenVINO GPU Custom Layer Compatibility
// ═══════════════════════════════════════════════════════════════════════════

/**
 * KNNPointsSingle Operation (GPU-compatible single output)
 *
 * Same as KNNPoints but packs both outputs into one tensor.
 *
 * Inputs:
 *   0: p1 - [B, N1, 3] query points
 *   1: p2 - [B, N2, 3] source points
 *
 * Outputs:
 *   0: combined - [B, N1, K*2] where:
 *      - [:, :, :K] = squared distances
 *      - [:, :, K:] = indices as float (cast to int in post-processing)
 */
class KNNPointsSingle : public ov::op::Op {
public:
    OPENVINO_OP("KNNPointsSingle", "hggd");

    KNNPointsSingle() = default;

    KNNPointsSingle(const ov::Output<ov::Node>& p1,
                    const ov::Output<ov::Node>& p2,
                    int64_t k = 12);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }

private:
    int64_t m_k{12};
};


/**
 * BallQuerySingle Operation (GPU-compatible single output)
 *
 * Same as BallQuery but packs both outputs into one tensor.
 *
 * Inputs:
 *   0: p1 - [B, N1, 3] query points
 *   1: p2 - [B, N2, 3] source points
 *
 * Outputs:
 *   0: combined - [B, N1, K*2] where:
 *      - [:, :, :K] = squared distances (-1 for invalid)
 *      - [:, :, K:] = indices as float (-1 for invalid)
 */
class BallQuerySingle : public ov::op::Op {
public:
    OPENVINO_OP("BallQuerySingle", "hggd");

    BallQuerySingle() = default;

    BallQuerySingle(const ov::Output<ov::Node>& p1,
                    const ov::Output<ov::Node>& p2,
                    int64_t k = 32,
                    float radius = 0.2f);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }
    float get_radius() const { return m_radius; }

private:
    int64_t m_k{32};
    float m_radius{0.2f};
};


/**
 * FPSSingle Operation (GPU-compatible single output)
 *
 * Same as FPS but packs both outputs into one tensor.
 *
 * Inputs:
 *   0: points - [B, N, 3] input points
 *
 * Outputs:
 *   0: combined - [B, K, 4] where:
 *      - [:, :, :3] = sampled point xyz
 *      - [:, :, 3]  = sampled index as float
 */
class FPSSingle : public ov::op::Op {
public:
    OPENVINO_OP("FPSSingle", "hggd");

    FPSSingle() = default;

    FPSSingle(const ov::Output<ov::Node>& points,
              int64_t k = 1024);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }

private:
    int64_t m_k{1024};
};

/**
 * FPSWithLengths Operation (GPU-compatible with variable-length batches)
 *
 * Same as FPSSingle but with per-batch lengths for variable-length support.
 *
 * Inputs:
 *   0: points  - [B, N, 3] input points (may be zero-padded)
 *   1: lengths - [B] actual valid length for each batch element
 *
 * Outputs:
 *   0: combined - [B, K, 4] where:
 *      - [:, :, :3] = sampled point xyz
 *      - [:, :, 3]  = sampled index as float
 */
class FPSWithLengths : public ov::op::Op {
public:
    OPENVINO_OP("FPSWithLengths", "hggd");

    FPSWithLengths() = default;

    FPSWithLengths(const ov::Output<ov::Node>& points,
                   const ov::Output<ov::Node>& lengths,
                   int64_t k = 1024);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(
        const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override { return true; }

    int64_t get_k() const { return m_k; }

private:
    int64_t m_k{1024};
};

}  // namespace HGGDExtension
