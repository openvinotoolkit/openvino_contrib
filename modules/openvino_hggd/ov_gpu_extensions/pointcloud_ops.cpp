/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
// Point Cloud Operations Implementation for OpenVINO
// CPU fallback implementations for all point cloud operations

#include "pointcloud_ops.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace HGGDExtension {

// ═══════════════════════════════════════════════════════════════════════════
// KNNPoints Implementation
// ═══════════════════════════════════════════════════════════════════════════

KNNPoints::KNNPoints(const ov::Output<ov::Node>& p1,
                     const ov::Output<ov::Node>& p2,
                     int64_t k)
    : Op({p1, p2}), m_k(k) {
    constructor_validate_and_infer_types();
}

void KNNPoints::validate_and_infer_types() {
    const auto& p1_shape = get_input_partial_shape(0);
    const auto& p2_shape = get_input_partial_shape(1);

    // Expect [B, N1, 3] and [B, N2, 3]
    NODE_VALIDATION_CHECK(this, p1_shape.rank().is_static() && p1_shape.rank().get_length() == 3,
                          "p1 must be 3D [B, N1, 3]");
    NODE_VALIDATION_CHECK(this, p2_shape.rank().is_static() && p2_shape.rank().get_length() == 3,
                          "p2 must be 3D [B, N2, 3]");

    // Output shapes: [B, N1, K]
    ov::PartialShape out_shape{p1_shape[0], p1_shape[1], m_k};
    set_output_type(0, ov::element::f32, out_shape);  // dists
    set_output_type(1, ov::element::i32, out_shape);  // idx
}

std::shared_ptr<ov::Node> KNNPoints::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<KNNPoints>(new_args.at(0), new_args.at(1), m_k);
}

bool KNNPoints::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    return true;
}

bool KNNPoints::evaluate(ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) const {
    const float* p1 = inputs[0].data<float>();
    const float* p2 = inputs[1].data<float>();
    float* dists = outputs[0].data<float>();
    int32_t* idx = outputs[1].data<int32_t>();

    const auto& p1_shape = inputs[0].get_shape();
    const auto& p2_shape = inputs[1].get_shape();
    const int64_t B = p1_shape[0];
    const int64_t N1 = p1_shape[1];
    const int64_t N2 = p2_shape[1];
    const int64_t K = m_k;

    // For each batch
    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* p1_b = p1 + b * N1 * 3;
        const float* p2_b = p2 + b * N2 * 3;
        float* dists_b = dists + b * N1 * K;
        int32_t* idx_b = idx + b * N1 * K;

        // Temporary storage for distances
        std::vector<std::pair<float, int32_t>> dist_idx(N2);

        for (int64_t i = 0; i < N1; ++i) {
            const float qx = p1_b[i * 3 + 0];
            const float qy = p1_b[i * 3 + 1];
            const float qz = p1_b[i * 3 + 2];

            // Compute distances to all points
            for (int64_t j = 0; j < N2; ++j) {
                const float dx = p2_b[j * 3 + 0] - qx;
                const float dy = p2_b[j * 3 + 1] - qy;
                const float dz = p2_b[j * 3 + 2] - qz;
                dist_idx[j] = {dx*dx + dy*dy + dz*dz, static_cast<int32_t>(j)};
            }

            // Partial sort to get K smallest
            std::partial_sort(dist_idx.begin(), dist_idx.begin() + K, dist_idx.end());

            // Store results
            for (int64_t k = 0; k < K; ++k) {
                dists_b[i * K + k] = dist_idx[k].first;
                idx_b[i * K + k] = dist_idx[k].second;
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// BallQuery Implementation
// ═══════════════════════════════════════════════════════════════════════════

BallQuery::BallQuery(const ov::Output<ov::Node>& p1,
                     const ov::Output<ov::Node>& p2,
                     int64_t k,
                     float radius)
    : Op({p1, p2}), m_k(k), m_radius(radius) {
    constructor_validate_and_infer_types();
}

void BallQuery::validate_and_infer_types() {
    const auto& p1_shape = get_input_partial_shape(0);
    const auto& p2_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, p1_shape.rank().is_static() && p1_shape.rank().get_length() == 3,
                          "p1 must be 3D [B, N1, 3]");
    NODE_VALIDATION_CHECK(this, p2_shape.rank().is_static() && p2_shape.rank().get_length() == 3,
                          "p2 must be 3D [B, N2, 3]");

    ov::PartialShape out_shape{p1_shape[0], p1_shape[1], m_k};
    set_output_type(0, ov::element::f32, out_shape);  // dists
    set_output_type(1, ov::element::i32, out_shape);  // idx
}

std::shared_ptr<ov::Node> BallQuery::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<BallQuery>(new_args.at(0), new_args.at(1), m_k, m_radius);
}

bool BallQuery::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    visitor.on_attribute("radius", m_radius);
    return true;
}

bool BallQuery::evaluate(ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) const {
    const float* p1 = inputs[0].data<float>();
    const float* p2 = inputs[1].data<float>();
    float* dists = outputs[0].data<float>();
    int32_t* idx = outputs[1].data<int32_t>();

    const auto& p1_shape = inputs[0].get_shape();
    const auto& p2_shape = inputs[1].get_shape();
    const int64_t B = p1_shape[0];
    const int64_t N1 = p1_shape[1];
    const int64_t N2 = p2_shape[1];
    const int64_t K = m_k;
    const float radius_sq = m_radius * m_radius;

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* p1_b = p1 + b * N1 * 3;
        const float* p2_b = p2 + b * N2 * 3;
        float* dists_b = dists + b * N1 * K;
        int32_t* idx_b = idx + b * N1 * K;

        std::vector<std::pair<float, int32_t>> neighbors;
        neighbors.reserve(N2);

        for (int64_t i = 0; i < N1; ++i) {
            const float qx = p1_b[i * 3 + 0];
            const float qy = p1_b[i * 3 + 1];
            const float qz = p1_b[i * 3 + 2];

            neighbors.clear();

            // Find all points within radius
            for (int64_t j = 0; j < N2; ++j) {
                const float dx = p2_b[j * 3 + 0] - qx;
                const float dy = p2_b[j * 3 + 1] - qy;
                const float dz = p2_b[j * 3 + 2] - qz;
                const float d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < radius_sq) {
                    neighbors.emplace_back(d2, static_cast<int32_t>(j));
                }
            }

            // Sort by distance
            std::sort(neighbors.begin(), neighbors.end());

            // Fill output - use -1 for empty slots
            const int64_t valid_count = std::min(static_cast<int64_t>(neighbors.size()), K);
            for (int64_t k = 0; k < valid_count; ++k) {
                dists_b[i * K + k] = neighbors[k].first;
                idx_b[i * K + k] = neighbors[k].second;
            }
            // Fill remaining slots with -1
            for (int64_t k = valid_count; k < K; ++k) {
                dists_b[i * K + k] = -1.0f;
                idx_b[i * K + k] = -1;
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// FPS Implementation
// ═══════════════════════════════════════════════════════════════════════════

FPS::FPS(const ov::Output<ov::Node>& points, int64_t k)
    : Op({points}), m_k(k) {
    constructor_validate_and_infer_types();
}

void FPS::validate_and_infer_types() {
    const auto& pts_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this, pts_shape.rank().is_static() && pts_shape.rank().get_length() == 3,
                          "points must be 3D [B, N, 3]");

    // Output: sampled [B, K, 3], idx [B, K]
    set_output_type(0, ov::element::f32, ov::PartialShape{pts_shape[0], m_k, 3});
    set_output_type(1, ov::element::i32, ov::PartialShape{pts_shape[0], m_k});
}

std::shared_ptr<ov::Node> FPS::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<FPS>(new_args.at(0), m_k);
}

bool FPS::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    return true;
}

bool FPS::evaluate(ov::TensorVector& outputs,
                   const ov::TensorVector& inputs) const {
    const float* points = inputs[0].data<float>();
    float* sampled = outputs[0].data<float>();
    int32_t* idx = outputs[1].data<int32_t>();

    const auto& pts_shape = inputs[0].get_shape();
    const int64_t B = pts_shape[0];
    const int64_t N = pts_shape[1];
    const int64_t K = m_k;

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* pts_b = points + b * N * 3;
        float* samp_b = sampled + b * K * 3;
        int32_t* idx_b = idx + b * K;

        std::vector<float> min_dists(N, std::numeric_limits<float>::max());
        int64_t current = 0;  // Start with first point

        for (int64_t k = 0; k < K; ++k) {
            // Store current sample
            idx_b[k] = static_cast<int32_t>(current);
            samp_b[k * 3 + 0] = pts_b[current * 3 + 0];
            samp_b[k * 3 + 1] = pts_b[current * 3 + 1];
            samp_b[k * 3 + 2] = pts_b[current * 3 + 2];

            if (k == K - 1) break;

            const float cx = pts_b[current * 3 + 0];
            const float cy = pts_b[current * 3 + 1];
            const float cz = pts_b[current * 3 + 2];

            // Update distances and find farthest
            float max_dist = -1.0f;
            int64_t farthest = 0;

            for (int64_t i = 0; i < N; ++i) {
                const float dx = pts_b[i * 3 + 0] - cx;
                const float dy = pts_b[i * 3 + 1] - cy;
                const float dz = pts_b[i * 3 + 2] - cz;
                const float d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < min_dists[i]) {
                    min_dists[i] = d2;
                }

                if (min_dists[i] > max_dist) {
                    max_dist = min_dists[i];
                    farthest = i;
                }
            }

            current = farthest;
            min_dists[current] = -1.0f;  // Mark as sampled
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// MaskedGather Implementation
// ═══════════════════════════════════════════════════════════════════════════

MaskedGather::MaskedGather(const ov::Output<ov::Node>& points,
                           const ov::Output<ov::Node>& idx)
    : Op({points, idx}) {
    constructor_validate_and_infer_types();
}

void MaskedGather::validate_and_infer_types() {
    const auto& pts_shape = get_input_partial_shape(0);
    const auto& idx_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, pts_shape.rank().is_static() && pts_shape.rank().get_length() == 3,
                          "points must be 3D [B, N, D]");

    // Output shape depends on idx shape
    if (idx_shape.rank().is_static() && idx_shape.rank().get_length() == 3) {
        // [B, M, K] -> [B, M, K, D]
        set_output_type(0, ov::element::f32, 
                        ov::PartialShape{idx_shape[0], idx_shape[1], idx_shape[2], pts_shape[2]});
    } else if (idx_shape.rank().is_static() && idx_shape.rank().get_length() == 2) {
        // [B, K] -> [B, K, D]
        set_output_type(0, ov::element::f32,
                        ov::PartialShape{idx_shape[0], idx_shape[1], pts_shape[2]});
    } else {
        set_output_type(0, ov::element::f32, ov::PartialShape::dynamic());
    }
}

std::shared_ptr<ov::Node> MaskedGather::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<MaskedGather>(new_args.at(0), new_args.at(1));
}

bool MaskedGather::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

bool MaskedGather::evaluate(ov::TensorVector& outputs,
                            const ov::TensorVector& inputs) const {
    const float* points = inputs[0].data<float>();
    const int32_t* idx = inputs[1].data<int32_t>();
    float* output = outputs[0].data<float>();

    const auto& pts_shape = inputs[0].get_shape();
    const auto& idx_shape = inputs[1].get_shape();
    const int64_t B = pts_shape[0];
    const int64_t N = pts_shape[1];
    const int64_t D = pts_shape[2];

    if (idx_shape.size() == 3) {
        // [B, M, K] indices
        const int64_t M = idx_shape[1];
        const int64_t K = idx_shape[2];

        #pragma omp parallel for collapse(3)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t k = 0; k < K; ++k) {
                    const int64_t idx_offset = b * M * K + m * K + k;
                    const int32_t src_idx = idx[idx_offset];
                    const int64_t out_offset = idx_offset * D;

                    if (src_idx >= 0 && src_idx < N) {
                        const int64_t pts_offset = b * N * D + src_idx * D;
                        for (int64_t d = 0; d < D; ++d) {
                            output[out_offset + d] = points[pts_offset + d];
                        }
                    } else {
                        for (int64_t d = 0; d < D; ++d) {
                            output[out_offset + d] = 0.0f;
                        }
                    }
                }
            }
        }
    } else {
        // [B, K] indices
        const int64_t K = idx_shape[1];

        #pragma omp parallel for collapse(2)
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t k = 0; k < K; ++k) {
                const int32_t src_idx = idx[b * K + k];
                const int64_t out_offset = (b * K + k) * D;

                if (src_idx >= 0 && src_idx < N) {
                    const int64_t pts_offset = b * N * D + src_idx * D;
                    for (int64_t d = 0; d < D; ++d) {
                        output[out_offset + d] = points[pts_offset + d];
                    }
                } else {
                    for (int64_t d = 0; d < D; ++d) {
                        output[out_offset + d] = 0.0f;
                    }
                }
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// GatherMaxPool Implementation
// ═══════════════════════════════════════════════════════════════════════════

GatherMaxPool::GatherMaxPool(const ov::Output<ov::Node>& points,
                             const ov::Output<ov::Node>& idx)
    : Op({points, idx}) {
    constructor_validate_and_infer_types();
}

void GatherMaxPool::validate_and_infer_types() {
    const auto& pts_shape = get_input_partial_shape(0);
    const auto& idx_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, pts_shape.rank().is_static() && pts_shape.rank().get_length() == 3,
                          "points must be 3D [B, N, D]");
    NODE_VALIDATION_CHECK(this, idx_shape.rank().is_static() && idx_shape.rank().get_length() == 3,
                          "idx must be 3D [B, M, K]");

    // Output: [B, M, D]
    set_output_type(0, ov::element::f32, 
                    ov::PartialShape{idx_shape[0], idx_shape[1], pts_shape[2]});
}

std::shared_ptr<ov::Node> GatherMaxPool::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<GatherMaxPool>(new_args.at(0), new_args.at(1));
}

bool GatherMaxPool::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

bool GatherMaxPool::evaluate(ov::TensorVector& outputs,
                             const ov::TensorVector& inputs) const {
    const float* points = inputs[0].data<float>();
    const int32_t* idx = inputs[1].data<int32_t>();
    float* output = outputs[0].data<float>();

    const auto& pts_shape = inputs[0].get_shape();
    const auto& idx_shape = inputs[1].get_shape();
    const int64_t B = pts_shape[0];
    const int64_t N = pts_shape[1];
    const int64_t D = pts_shape[2];
    const int64_t M = idx_shape[1];
    const int64_t K = idx_shape[2];

    constexpr float NEG_INF = -1e30f;

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            const int64_t out_offset = (b * M + m) * D;

            // Initialize with -inf
            for (int64_t d = 0; d < D; ++d) {
                output[out_offset + d] = NEG_INF;
            }

            // Max over K neighbors
            for (int64_t k = 0; k < K; ++k) {
                const int32_t src_idx = idx[b * M * K + m * K + k];

                if (src_idx >= 0 && src_idx < N) {
                    const int64_t pts_offset = b * N * D + src_idx * D;
                    for (int64_t d = 0; d < D; ++d) {
                        const float val = points[pts_offset + d];
                        if (val > output[out_offset + d]) {
                            output[out_offset + d] = val;
                        }
                    }
                }
            }

            // Replace -inf with 0 for empty groups
            for (int64_t d = 0; d < D; ++d) {
                if (output[out_offset + d] < -1e29f) {
                    output[out_offset + d] = 0.0f;
                }
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// PointGather Implementation
// ═══════════════════════════════════════════════════════════════════════════

PointGather::PointGather(const ov::Output<ov::Node>& points,
                         const ov::Output<ov::Node>& idx)
    : Op({points, idx}) {
    constructor_validate_and_infer_types();
}

void PointGather::validate_and_infer_types() {
    const auto& pts_shape = get_input_partial_shape(0);
    const auto& idx_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, pts_shape.rank().is_static() && pts_shape.rank().get_length() == 3,
                          "points must be 3D [B, N, D]");
    NODE_VALIDATION_CHECK(this, idx_shape.rank().is_static() && idx_shape.rank().get_length() == 2,
                          "idx must be 2D [B, K]");

    // Output: [B, K, D]
    set_output_type(0, ov::element::f32,
                    ov::PartialShape{idx_shape[0], idx_shape[1], pts_shape[2]});
}

std::shared_ptr<ov::Node> PointGather::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<PointGather>(new_args.at(0), new_args.at(1));
}

bool PointGather::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

bool PointGather::evaluate(ov::TensorVector& outputs,
                           const ov::TensorVector& inputs) const {
    const float* points = inputs[0].data<float>();
    const int32_t* idx = inputs[1].data<int32_t>();
    float* output = outputs[0].data<float>();

    const auto& pts_shape = inputs[0].get_shape();
    const auto& idx_shape = inputs[1].get_shape();
    const int64_t B = pts_shape[0];
    const int64_t N = pts_shape[1];
    const int64_t D = pts_shape[2];
    const int64_t K = idx_shape[1];

    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t k = 0; k < K; ++k) {
            const int32_t src_idx = idx[b * K + k];
            const int64_t out_offset = (b * K + k) * D;

            if (src_idx >= 0 && src_idx < N) {
                const int64_t pts_offset = b * N * D + src_idx * D;
                for (int64_t d = 0; d < D; ++d) {
                    output[out_offset + d] = points[pts_offset + d];
                }
            } else {
                for (int64_t d = 0; d < D; ++d) {
                    output[out_offset + d] = 0.0f;
                }
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// KNNPointsSingle Implementation (GPU-compatible single output)
// ═══════════════════════════════════════════════════════════════════════════

KNNPointsSingle::KNNPointsSingle(const ov::Output<ov::Node>& p1,
                                 const ov::Output<ov::Node>& p2,
                                 int64_t k)
    : Op({p1, p2}), m_k(k) {
    constructor_validate_and_infer_types();
}

void KNNPointsSingle::validate_and_infer_types() {
    const auto& p1_shape = get_input_partial_shape(0);
    const auto& p2_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, p1_shape.rank().is_static() && p1_shape.rank().get_length() == 3,
                          "p1 must be 3D [B, N1, 3]");
    NODE_VALIDATION_CHECK(this, p2_shape.rank().is_static() && p2_shape.rank().get_length() == 3,
                          "p2 must be 3D [B, N2, 3]");

    // Output shape: [B, N1, K*2] - first K are dists, next K are indices
    ov::PartialShape out_shape{p1_shape[0], p1_shape[1], m_k * 2};
    set_output_type(0, ov::element::f32, out_shape);
}

std::shared_ptr<ov::Node> KNNPointsSingle::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<KNNPointsSingle>(new_args.at(0), new_args.at(1), m_k);
}

bool KNNPointsSingle::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    return true;
}

bool KNNPointsSingle::evaluate(ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) const {
    const float* p1 = inputs[0].data<float>();
    const float* p2 = inputs[1].data<float>();
    float* combined = outputs[0].data<float>();

    const auto& p1_shape = inputs[0].get_shape();
    const auto& p2_shape = inputs[1].get_shape();
    const int64_t B = p1_shape[0];
    const int64_t N1 = p1_shape[1];
    const int64_t N2 = p2_shape[1];
    const int64_t K = m_k;
    const int64_t K2 = K * 2;

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* p1_b = p1 + b * N1 * 3;
        const float* p2_b = p2 + b * N2 * 3;
        float* out_b = combined + b * N1 * K2;

        std::vector<std::pair<float, int32_t>> dist_idx(N2);

        for (int64_t i = 0; i < N1; ++i) {
            const float qx = p1_b[i * 3 + 0];
            const float qy = p1_b[i * 3 + 1];
            const float qz = p1_b[i * 3 + 2];

            for (int64_t j = 0; j < N2; ++j) {
                const float dx = p2_b[j * 3 + 0] - qx;
                const float dy = p2_b[j * 3 + 1] - qy;
                const float dz = p2_b[j * 3 + 2] - qz;
                dist_idx[j] = {dx*dx + dy*dy + dz*dz, static_cast<int32_t>(j)};
            }

            std::partial_sort(dist_idx.begin(), dist_idx.begin() + K, dist_idx.end());

            // Combined output: [dists_0..dists_K-1, idx_0..idx_K-1]
            for (int64_t k = 0; k < K; ++k) {
                out_b[i * K2 + k] = dist_idx[k].first;           // Distance
                out_b[i * K2 + K + k] = static_cast<float>(dist_idx[k].second);  // Index as float
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// BallQuerySingle Implementation (GPU-compatible single output)
// ═══════════════════════════════════════════════════════════════════════════

BallQuerySingle::BallQuerySingle(const ov::Output<ov::Node>& p1,
                                 const ov::Output<ov::Node>& p2,
                                 int64_t k,
                                 float radius)
    : Op({p1, p2}), m_k(k), m_radius(radius) {
    constructor_validate_and_infer_types();
}

void BallQuerySingle::validate_and_infer_types() {
    const auto& p1_shape = get_input_partial_shape(0);
    const auto& p2_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, p1_shape.rank().is_static() && p1_shape.rank().get_length() == 3,
                          "p1 must be 3D [B, N1, 3]");
    NODE_VALIDATION_CHECK(this, p2_shape.rank().is_static() && p2_shape.rank().get_length() == 3,
                          "p2 must be 3D [B, N2, 3]");

    // Output shape: [B, N1, K*2]
    ov::PartialShape out_shape{p1_shape[0], p1_shape[1], m_k * 2};
    set_output_type(0, ov::element::f32, out_shape);
}

std::shared_ptr<ov::Node> BallQuerySingle::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<BallQuerySingle>(new_args.at(0), new_args.at(1), m_k, m_radius);
}

bool BallQuerySingle::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    visitor.on_attribute("radius", m_radius);
    return true;
}

bool BallQuerySingle::evaluate(ov::TensorVector& outputs,
                               const ov::TensorVector& inputs) const {
    const float* p1 = inputs[0].data<float>();
    const float* p2 = inputs[1].data<float>();
    float* combined = outputs[0].data<float>();

    const auto& p1_shape = inputs[0].get_shape();
    const auto& p2_shape = inputs[1].get_shape();
    const int64_t B = p1_shape[0];
    const int64_t N1 = p1_shape[1];
    const int64_t N2 = p2_shape[1];
    const int64_t K = m_k;
    const int64_t K2 = K * 2;
    const float radius_sq = m_radius * m_radius;

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* p1_b = p1 + b * N1 * 3;
        const float* p2_b = p2 + b * N2 * 3;
        float* out_b = combined + b * N1 * K2;

        std::vector<std::pair<float, int32_t>> neighbors;
        neighbors.reserve(N2);

        for (int64_t i = 0; i < N1; ++i) {
            const float qx = p1_b[i * 3 + 0];
            const float qy = p1_b[i * 3 + 1];
            const float qz = p1_b[i * 3 + 2];

            neighbors.clear();

            for (int64_t j = 0; j < N2; ++j) {
                const float dx = p2_b[j * 3 + 0] - qx;
                const float dy = p2_b[j * 3 + 1] - qy;
                const float dz = p2_b[j * 3 + 2] - qz;
                const float d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < radius_sq) {
                    neighbors.emplace_back(d2, static_cast<int32_t>(j));
                }
            }

            std::sort(neighbors.begin(), neighbors.end());

            const int64_t valid_count = std::min(static_cast<int64_t>(neighbors.size()), K);
            
            // Combined output: [dists_0..dists_K-1, idx_0..idx_K-1]
            for (int64_t k = 0; k < valid_count; ++k) {
                out_b[i * K2 + k] = neighbors[k].first;
                out_b[i * K2 + K + k] = static_cast<float>(neighbors[k].second);
            }
            for (int64_t k = valid_count; k < K; ++k) {
                out_b[i * K2 + k] = -1.0f;
                out_b[i * K2 + K + k] = -1.0f;
            }
        }
    }

    return true;
}


// ═══════════════════════════════════════════════════════════════════════════
// FPSSingle Implementation (GPU-compatible single output)
// ═══════════════════════════════════════════════════════════════════════════

FPSSingle::FPSSingle(const ov::Output<ov::Node>& points, int64_t k)
    : Op({points}), m_k(k) {
    constructor_validate_and_infer_types();
}

void FPSSingle::validate_and_infer_types() {
    const auto& pts_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(this, pts_shape.rank().is_static() && pts_shape.rank().get_length() == 3,
                          "points must be 3D [B, N, 3]");

    // Output: [B, K, 4] - xyz + index
    set_output_type(0, ov::element::f32, ov::PartialShape{pts_shape[0], m_k, 4});
}

std::shared_ptr<ov::Node> FPSSingle::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<FPSSingle>(new_args.at(0), m_k);
}

bool FPSSingle::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    return true;
}

bool FPSSingle::evaluate(ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) const {
    const float* points = inputs[0].data<float>();
    float* combined = outputs[0].data<float>();

    const auto& pts_shape = inputs[0].get_shape();
    const int64_t B = pts_shape[0];
    const int64_t N = pts_shape[1];
    const int64_t K = m_k;

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* pts_b = points + b * N * 3;
        float* out_b = combined + b * K * 4;

        std::vector<float> min_dists(N, std::numeric_limits<float>::max());
        int64_t current = 0;

        for (int64_t k = 0; k < K; ++k) {
            // Store combined: xyz + index
            out_b[k * 4 + 0] = pts_b[current * 3 + 0];
            out_b[k * 4 + 1] = pts_b[current * 3 + 1];
            out_b[k * 4 + 2] = pts_b[current * 3 + 2];
            out_b[k * 4 + 3] = static_cast<float>(current);

            if (k == K - 1) break;

            const float cx = pts_b[current * 3 + 0];
            const float cy = pts_b[current * 3 + 1];
            const float cz = pts_b[current * 3 + 2];

            float max_dist = -1.0f;
            int64_t farthest = 0;

            for (int64_t i = 0; i < N; ++i) {
                const float dx = pts_b[i * 3 + 0] - cx;
                const float dy = pts_b[i * 3 + 1] - cy;
                const float dz = pts_b[i * 3 + 2] - cz;
                const float d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < min_dists[i]) {
                    min_dists[i] = d2;
                }

                if (min_dists[i] > max_dist) {
                    max_dist = min_dists[i];
                    farthest = i;
                }
            }

            current = farthest;
            min_dists[current] = -1.0f;
        }
    }

    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// FPSWithLengths - GPU-compatible with variable-length batch support
// ════════════════════════════════════════════════════════════════════════════

FPSWithLengths::FPSWithLengths(const ov::Output<ov::Node>& points,
                               const ov::Output<ov::Node>& lengths,
                               int64_t k)
    : Op({points, lengths}), m_k(k) {
    constructor_validate_and_infer_types();
}

void FPSWithLengths::validate_and_infer_types() {
    const auto& pts_shape = get_input_partial_shape(0);
    const auto& len_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this, pts_shape.rank().is_static() && pts_shape.rank().get_length() == 3,
                          "points must be 3D [B, N, 3]");
    NODE_VALIDATION_CHECK(this, len_shape.rank().is_static() && len_shape.rank().get_length() == 1,
                          "lengths must be 1D [B]");

    // Output: [B, K, 4] - xyz + index
    set_output_type(0, ov::element::f32, ov::PartialShape{pts_shape[0], m_k, 4});
}

std::shared_ptr<ov::Node> FPSWithLengths::clone_with_new_inputs(
    const ov::OutputVector& new_args) const {
    return std::make_shared<FPSWithLengths>(new_args.at(0), new_args.at(1), m_k);
}

bool FPSWithLengths::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("k", m_k);
    return true;
}

bool FPSWithLengths::evaluate(ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) const {
    const float* points = inputs[0].data<float>();
    const float* lengths_f = inputs[1].data<float>();
    float* combined = outputs[0].data<float>();

    const auto& pts_shape = inputs[0].get_shape();
    const int64_t B = pts_shape[0];
    const int64_t N = pts_shape[1];
    const int64_t K = m_k;

    #pragma omp parallel for
    for (int64_t b = 0; b < B; ++b) {
        const float* pts_b = points + b * N * 3;
        float* out_b = combined + b * K * 4;
        const int64_t valid_len = static_cast<int64_t>(lengths_f[b]);
        const int64_t k_actual = std::min(K, valid_len);

        if (valid_len <= 0) {
            // No valid points - output zeros
            for (int64_t k = 0; k < K; ++k) {
                out_b[k * 4 + 0] = 0.0f;
                out_b[k * 4 + 1] = 0.0f;
                out_b[k * 4 + 2] = 0.0f;
                out_b[k * 4 + 3] = 0.0f;
            }
            continue;
        }

        std::vector<float> min_dists(valid_len, std::numeric_limits<float>::max());
        int64_t current = 0;

        for (int64_t k = 0; k < k_actual; ++k) {
            // Store combined: xyz + index
            out_b[k * 4 + 0] = pts_b[current * 3 + 0];
            out_b[k * 4 + 1] = pts_b[current * 3 + 1];
            out_b[k * 4 + 2] = pts_b[current * 3 + 2];
            out_b[k * 4 + 3] = static_cast<float>(current);

            if (k == k_actual - 1) break;

            const float cx = pts_b[current * 3 + 0];
            const float cy = pts_b[current * 3 + 1];
            const float cz = pts_b[current * 3 + 2];

            float max_dist = -1.0f;
            int64_t farthest = 0;

            // Only iterate over valid points
            for (int64_t i = 0; i < valid_len; ++i) {
                const float dx = pts_b[i * 3 + 0] - cx;
                const float dy = pts_b[i * 3 + 1] - cy;
                const float dz = pts_b[i * 3 + 2] - cz;
                const float d2 = dx*dx + dy*dy + dz*dz;

                if (d2 < min_dists[i]) {
                    min_dists[i] = d2;
                }

                if (min_dists[i] > max_dist) {
                    max_dist = min_dists[i];
                    farthest = i;
                }
            }

            current = farthest;
            min_dists[current] = -1.0f;
        }

        // Pad remaining with last valid point
        if (k_actual < K && k_actual > 0) {
            const float last_x = out_b[(k_actual - 1) * 4 + 0];
            const float last_y = out_b[(k_actual - 1) * 4 + 1];
            const float last_z = out_b[(k_actual - 1) * 4 + 2];
            const float last_idx = out_b[(k_actual - 1) * 4 + 3];
            for (int64_t k = k_actual; k < K; ++k) {
                out_b[k * 4 + 0] = last_x;
                out_b[k * 4 + 1] = last_y;
                out_b[k * 4 + 2] = last_z;
                out_b[k * 4 + 3] = last_idx;
            }
        }
    }

    return true;
}

}  // namespace HGGDExtension
