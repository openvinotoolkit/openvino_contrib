/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

// BEVPool Operation CPU Implementation for OpenVINO
// Reference fallback; GPU acceleration via OpenCL custom layer

#include "bev_pool_op.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

using namespace BEVFusionExtension;

BEVPool::BEVPool(const ov::Output<ov::Node>& depth_logits,
                 const ov::Output<ov::Node>& context_feats,
                 const ov::Output<ov::Node>& geom,
                 int64_t nx, int64_t ny, int64_t nz,
                 int64_t num_cams, int64_t depth_bins,
                 int64_t channels, int64_t feat_h, int64_t feat_w,
                 float x_min, float y_min, float z_min,
                 float x_step, float y_step, float z_step)
    : Op({depth_logits, context_feats, geom}),
      m_nx(nx), m_ny(ny), m_nz(nz),
      m_num_cams(num_cams), m_depth_bins(depth_bins),
      m_channels(channels), m_feat_h(feat_h), m_feat_w(feat_w),
      m_x_min(x_min), m_y_min(y_min), m_z_min(z_min),
      m_x_step(x_step), m_y_step(y_step), m_z_step(z_step)
{
    constructor_validate_and_infer_types();
}

void BEVPool::validate_and_infer_types() {
    // Output: [1, C*NZ, NX, NY]
    set_output_type(0, ov::element::f32,
                    ov::PartialShape{1, m_channels * m_nz, m_nx, m_ny});
}

std::shared_ptr<ov::Node> BEVPool::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 3, "BEVPool requires 3 inputs");
    return std::make_shared<BEVPool>(
        new_args[0], new_args[1], new_args[2],
        m_nx, m_ny, m_nz,
        m_num_cams, m_depth_bins, m_channels, m_feat_h, m_feat_w,
        m_x_min, m_y_min, m_z_min,
        m_x_step, m_y_step, m_z_step);
}

bool BEVPool::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("nx", m_nx);
    visitor.on_attribute("ny", m_ny);
    visitor.on_attribute("nz", m_nz);
    visitor.on_attribute("num_cams", m_num_cams);
    visitor.on_attribute("depth_bins", m_depth_bins);
    visitor.on_attribute("channels", m_channels);
    visitor.on_attribute("feat_h", m_feat_h);
    visitor.on_attribute("feat_w", m_feat_w);
    visitor.on_attribute("x_min", m_x_min);
    visitor.on_attribute("y_min", m_y_min);
    visitor.on_attribute("z_min", m_z_min);
    visitor.on_attribute("x_step", m_x_step);
    visitor.on_attribute("y_step", m_y_step);
    visitor.on_attribute("z_step", m_z_step);
    return true;
}

bool BEVPool::evaluate(ov::TensorVector& outputs,
                       const ov::TensorVector& inputs) const
{
    const auto& depth_t  = inputs[0];   // [N, D, H, W]
    const auto& ctx_t    = inputs[1];   // [N, C, H, W]
    const auto& geom_t   = inputs[2];   // [N*D*H*W, 3]

    auto dshape = depth_t.get_shape();
    int N = static_cast<int>(dshape[0]);
    int D = static_cast<int>(dshape[1]);
    int H = static_cast<int>(dshape[2]);
    int W = static_cast<int>(dshape[3]);
    int C_ch = static_cast<int>(ctx_t.get_shape()[1]);

    int total_points = N * D * H * W;
    int fH_fW = H * W;
    int nx = static_cast<int>(m_nx);
    int ny = static_cast<int>(m_ny);
    int nz = static_cast<int>(m_nz);
    int output_elems = C_ch * nz * nx * ny;

    // Output
    ov::Shape out_shape{1, static_cast<size_t>(C_ch * nz),
                        static_cast<size_t>(nx), static_cast<size_t>(ny)};
    outputs[0].set_shape(out_shape);
    float* out_ptr = outputs[0].data<float>();
    std::fill(out_ptr, out_ptr + output_elems, 0.0f);

    const float* dl = depth_t.data<float>();
    const float* cf = ctx_t.data<float>();
    const float* gm = geom_t.data<float>();

    // Pre-compute softmax over depth for all pixels
    std::vector<float> depth_sm(total_points);
    for (int cam = 0; cam < N; ++cam) {
        for (int hw = 0; hw < fH_fW; ++hw) {
            int base = cam * D * fH_fW + hw;
            float mx = -1e30f;
            for (int d = 0; d < D; ++d) {
                float v = dl[base + d * fH_fW];
                if (v > mx) mx = v;
            }
            float se = 0.0f;
            for (int d = 0; d < D; ++d) {
                float e = std::exp(dl[base + d * fH_fW] - mx);
                depth_sm[cam * D * fH_fW + d * fH_fW + hw] = e;
                se += e;
            }
            float inv = 1.0f / se;
            for (int d = 0; d < D; ++d)
                depth_sm[cam * D * fH_fW + d * fH_fW + hw] *= inv;
        }
    }

    // Scatter weighted features to BEV grid
    int nx_ny = nx * ny;
    for (int p = 0; p < total_points; ++p) {
        float gx = gm[p * 3 + 0];
        float gy = gm[p * 3 + 1];
        float gz = gm[p * 3 + 2];
        int ix = static_cast<int>((gx - m_x_min) / m_x_step);
        int iy = static_cast<int>((gy - m_y_min) / m_y_step);
        int iz = static_cast<int>((gz - m_z_min) / m_z_step);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
            continue;

        float dw = depth_sm[p];
        if (dw < 1e-6f) continue;

        int cam = p / (D * fH_fW);
        int hw = p % fH_fW;
        int ctx_base = cam * C_ch * fH_fW + hw;

        for (int c = 0; c < C_ch; ++c) {
            float val = dw * cf[ctx_base + c * fH_fW];
            int oidx = (c * nz + iz) * nx_ny + ix * ny + iy;
            out_ptr[oidx] += val;
        }
    }
    return true;
}

bool BEVPool::has_evaluate() const { return true; }


// ═══════════════════════════════════════════════════════════════════════════
// BEVPoolScatter: int32 fixed-point scatter (pre-computed softmax input)
// ═══════════════════════════════════════════════════════════════════════════

BEVPoolScatter::BEVPoolScatter(
    const ov::Output<ov::Node>& depth_probs,
    const ov::Output<ov::Node>& context_feats,
    const ov::Output<ov::Node>& geom,
    int64_t nx, int64_t ny, int64_t nz,
    int64_t num_cams, int64_t depth_bins,
    int64_t channels, int64_t feat_h, int64_t feat_w,
    float x_min, float y_min, float z_min,
    float x_step, float y_step, float z_step)
    : Op({depth_probs, context_feats, geom}),
      m_nx(nx), m_ny(ny), m_nz(nz),
      m_num_cams(num_cams), m_depth_bins(depth_bins),
      m_channels(channels), m_feat_h(feat_h), m_feat_w(feat_w),
      m_x_min(x_min), m_y_min(y_min), m_z_min(z_min),
      m_x_step(x_step), m_y_step(y_step), m_z_step(z_step)
{
    constructor_validate_and_infer_types();
}

void BEVPoolScatter::validate_and_infer_types() {
    // Output: int32 [1, C*NZ, NX, NY]
    set_output_type(0, ov::element::i32,
                    ov::PartialShape{1, m_channels * m_nz, m_nx, m_ny});
}

std::shared_ptr<ov::Node> BEVPoolScatter::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 3, "BEVPoolScatter requires 3 inputs");
    return std::make_shared<BEVPoolScatter>(
        new_args[0], new_args[1], new_args[2],
        m_nx, m_ny, m_nz,
        m_num_cams, m_depth_bins, m_channels, m_feat_h, m_feat_w,
        m_x_min, m_y_min, m_z_min,
        m_x_step, m_y_step, m_z_step);
}

bool BEVPoolScatter::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("nx", m_nx);
    visitor.on_attribute("ny", m_ny);
    visitor.on_attribute("nz", m_nz);
    visitor.on_attribute("num_cams", m_num_cams);
    visitor.on_attribute("depth_bins", m_depth_bins);
    visitor.on_attribute("channels", m_channels);
    visitor.on_attribute("feat_h", m_feat_h);
    visitor.on_attribute("feat_w", m_feat_w);
    visitor.on_attribute("x_min", m_x_min);
    visitor.on_attribute("y_min", m_y_min);
    visitor.on_attribute("z_min", m_z_min);
    visitor.on_attribute("x_step", m_x_step);
    visitor.on_attribute("y_step", m_y_step);
    visitor.on_attribute("z_step", m_z_step);
    return true;
}

bool BEVPoolScatter::evaluate(ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) const
{
    const float* dp = inputs[0].data<float>();   // depth_probs
    const float* cf = inputs[1].data<float>();   // context_feats
    const float* gm = inputs[2].data<float>();   // geom

    auto shape = inputs[0].get_shape();
    int N = shape[0], D = shape[1], H = shape[2], W = shape[3];
    int C = static_cast<int>(inputs[1].get_shape()[1]);
    int total_pts = N * D * H * W;
    int HW = H * W;
    int nx = m_nx, ny = m_ny, nz = m_nz;
    int nx_ny = nx * ny;
    int output_elems = C * nz * nx * ny;

    constexpr int FP_SCALE = 8192;

    ov::Shape out_shape{1, size_t(C * nz), size_t(nx), size_t(ny)};
    outputs[0].set_shape(out_shape);
    int32_t* out = outputs[0].data<int32_t>();
    std::fill(out, out + output_elems, 0);

    for (int p = 0; p < total_pts; ++p) {
        float gx = gm[p * 3], gy = gm[p * 3 + 1], gz = gm[p * 3 + 2];
        int ix = int((gx - m_x_min) / m_x_step);
        int iy = int((gy - m_y_min) / m_y_step);
        int iz = int((gz - m_z_min) / m_z_step);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz)
            continue;

        float dw = dp[p];
        if (dw < 1e-6f) continue;

        int cam = p / (D * HW);
        int hw = p % HW;
        int ctx_base = cam * C * HW + hw;

        for (int c = 0; c < C; ++c) {
            float val = dw * cf[ctx_base + c * HW];
            int ival = int(val * FP_SCALE);
            int oidx = (c * nz + iz) * nx_ny + ix * ny + iy;
            out[oidx] += ival;
        }
    }
    return true;
}

bool BEVPoolScatter::has_evaluate() const { return true; }


// ═══════════════════════════════════════════════════════════════════════════
// BEVPoolConvert: int32 fixed-point → float32
// ═══════════════════════════════════════════════════════════════════════════

BEVPoolConvert::BEVPoolConvert(
    const ov::Output<ov::Node>& bev_accum, int64_t scale)
    : Op({bev_accum}), m_scale(scale)
{
    constructor_validate_and_infer_types();
}

void BEVPoolConvert::validate_and_infer_types() {
    set_output_type(0, ov::element::f32, get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> BEVPoolConvert::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 1, "BEVPoolConvert requires 1 input");
    return std::make_shared<BEVPoolConvert>(new_args[0], m_scale);
}

bool BEVPoolConvert::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("scale", m_scale);
    return true;
}

bool BEVPoolConvert::evaluate(ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) const
{
    auto shape = inputs[0].get_shape();
    outputs[0].set_shape(shape);
    const int32_t* in = inputs[0].data<int32_t>();
    float* out = outputs[0].data<float>();
    size_t n = inputs[0].get_size();
    float inv = 1.0f / float(m_scale);
    for (size_t i = 0; i < n; ++i)
        out[i] = float(in[i]) * inv;
    return true;
}

bool BEVPoolConvert::has_evaluate() const { return true; }


// ═══════════════════════════════════════════════════════════════════════════
// BEVPoolBinSort: GPU counting sort of geometry → sorted indices + intervals
// ═══════════════════════════════════════════════════════════════════════════

BEVPoolBinSort::BEVPoolBinSort(
    const ov::Output<ov::Node>& geom,
    int64_t nx, int64_t ny, int64_t total_pts,
    float x_min, float y_min, float x_step, float y_step)
    : Op({geom}),
      m_nx(nx), m_ny(ny), m_total_pts(total_pts),
      m_x_min(x_min), m_y_min(y_min), m_x_step(x_step), m_y_step(y_step)
{
    constructor_validate_and_infer_types();
}

void BEVPoolBinSort::validate_and_infer_types() {
    int64_t n_cells = m_nx * m_ny;
    int64_t packed_size = m_total_pts * 2 + n_cells * 2;
    set_output_type(0, ov::element::i32, ov::PartialShape{packed_size});
}

std::shared_ptr<ov::Node> BEVPoolBinSort::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 1, "BEVPoolBinSort requires 1 input");
    return std::make_shared<BEVPoolBinSort>(
        new_args[0], m_nx, m_ny, m_total_pts,
        m_x_min, m_y_min, m_x_step, m_y_step);
}

bool BEVPoolBinSort::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("nx", m_nx);
    visitor.on_attribute("ny", m_ny);
    visitor.on_attribute("total_pts", m_total_pts);
    visitor.on_attribute("x_min", m_x_min);
    visitor.on_attribute("y_min", m_y_min);
    visitor.on_attribute("x_step", m_x_step);
    visitor.on_attribute("y_step", m_y_step);
    return true;
}

bool BEVPoolBinSort::evaluate(ov::TensorVector& outputs,
                              const ov::TensorVector& inputs) const
{
    const float* gm = inputs[0].data<float>();
    int nx = static_cast<int>(m_nx);
    int ny = static_cast<int>(m_ny);
    int total = static_cast<int>(m_total_pts);
    int n_cells = nx * ny;
    int packed_size = total * 2 + n_cells * 2;

    outputs[0].set_shape({size_t(packed_size)});
    int32_t* packed = outputs[0].data<int32_t>();

    // Packed layout: [sorted_ranks | cell_scratch | interval_starts | interval_lengths]
    int32_t* sorted   = packed;
    int32_t* cells    = packed + total;
    int32_t* starts   = packed + 2 * total;
    int32_t* lengths  = packed + 2 * total + n_cells;

    // Phase 1: compute cell per point + count
    std::fill(lengths, lengths + n_cells, 0);
    for (int p = 0; p < total; ++p) {
        float gx = gm[p * 3], gy = gm[p * 3 + 1];
        int ix = static_cast<int>(std::floor((gx - m_x_min) / m_x_step));
        int iy = static_cast<int>(std::floor((gy - m_y_min) / m_y_step));
        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny) {
            int cell = ix * ny + iy;
            cells[p] = cell;
            lengths[cell]++;
        } else {
            cells[p] = -1;
        }
    }

    // Phase 2: prefix sum
    starts[0] = 0;
    for (int i = 1; i < n_cells; ++i)
        starts[i] = starts[i - 1] + lengths[i - 1];

    // Phase 3: scatter
    std::vector<int32_t> write_pos(starts, starts + n_cells);
    for (int p = 0; p < total; ++p) {
        int cell = cells[p];
        if (cell >= 0)
            sorted[write_pos[cell]++] = p;
    }
    return true;
}

bool BEVPoolBinSort::has_evaluate() const { return true; }


// ═══════════════════════════════════════════════════════════════════════════
// BEVPoolV2: pre-sorted interval-based scatter (no atomics)
// ═══════════════════════════════════════════════════════════════════════════

BEVPoolV2::BEVPoolV2(
    const ov::Output<ov::Node>& depth_probs,
    const ov::Output<ov::Node>& context_feats,
    const ov::Output<ov::Node>& packed_sort,
    int64_t nx, int64_t ny, int64_t channels, int64_t feat_hw,
    int64_t depth_hw, int64_t total_pts)
    : Op({depth_probs, context_feats, packed_sort}),
      m_nx(nx), m_ny(ny), m_channels(channels), m_feat_hw(feat_hw),
      m_depth_hw(depth_hw), m_total_pts(total_pts)
{
    constructor_validate_and_infer_types();
}

void BEVPoolV2::validate_and_infer_types() {
    set_output_type(0, ov::element::f32,
                    ov::PartialShape{1, m_channels, m_nx, m_ny});
}

std::shared_ptr<ov::Node> BEVPoolV2::clone_with_new_inputs(
    const ov::OutputVector& new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 3, "BEVPoolV2 requires 3 inputs");
    return std::make_shared<BEVPoolV2>(
        new_args[0], new_args[1], new_args[2],
        m_nx, m_ny, m_channels, m_feat_hw, m_depth_hw, m_total_pts);
}

bool BEVPoolV2::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("nx", m_nx);
    visitor.on_attribute("ny", m_ny);
    visitor.on_attribute("channels", m_channels);
    visitor.on_attribute("feat_hw", m_feat_hw);
    visitor.on_attribute("depth_hw", m_depth_hw);
    visitor.on_attribute("total_pts", m_total_pts);
    return true;
}

bool BEVPoolV2::evaluate(ov::TensorVector& outputs,
                         const ov::TensorVector& inputs) const
{
    const float* dp        = inputs[0].data<float>();
    const float* cf        = inputs[1].data<float>();
    const int32_t* packed  = inputs[2].data<int32_t>();

    int total  = static_cast<int>(m_total_pts);
    int n_cells = static_cast<int>(m_nx * m_ny);
    int C_ch = static_cast<int>(m_channels);
    int HW = static_cast<int>(m_feat_hw);
    int D_HW = static_cast<int>(m_depth_hw);

    // Unpack: [sorted_ranks | cell_scratch | starts | lengths]
    const int32_t* ranks   = packed;
    const int32_t* starts  = packed + 2 * total;
    const int32_t* lengths = packed + 2 * total + n_cells;

    ov::Shape out_shape{1, size_t(C_ch), size_t(m_nx), size_t(m_ny)};
    outputs[0].set_shape(out_shape);
    float* out = outputs[0].data<float>();

    for (int cell = 0; cell < n_cells; ++cell) {
        int s = starts[cell];
        int l = lengths[cell];
        for (int c = 0; c < C_ch; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < l; ++i) {
                int pidx = ranks[s + i];
                int cam = pidx / D_HW;
                int hw = pidx % D_HW % HW;
                int feat_base = cam * C_ch * HW + hw;
                sum += dp[pidx] * cf[feat_base + c * HW];
            }
            out[c * n_cells + cell] = sum;
        }
    }
    return true;
}

bool BEVPoolV2::has_evaluate() const { return true; }
