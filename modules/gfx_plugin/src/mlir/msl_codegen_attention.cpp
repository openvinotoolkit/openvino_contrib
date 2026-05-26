// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_attention.hpp"

#include <cstring>
#include <sstream>
#include <utility>

#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

int32_t bitcast_f32_to_i32(float value) {
  int32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value),
                "GFX MSL float bitcast size mismatch");
  std::memcpy(&bits, &value, sizeof(value));
  return bits;
}

std::string metal_scalar_type(ov::element::Type type) {
  if (type == ov::element::f16) {
    return "half";
  }
  if (type == ov::element::f32) {
    return "float";
  }
  OPENVINO_THROW("GFX Metal SDPA: unsupported element type ", type);
}

std::string generate_msl_for_sdpa_variant(ov::element::Type type,
                                          bool has_mask) {
  const std::string scalar = metal_scalar_type(type);
  const std::string entry_point =
      has_mask ? "sdpa_kernel" : "sdpa_nomask_kernel";
  const uint32_t params_buffer = has_mask ? 4 : 3;
  const uint32_t output_buffer = has_mask ? 5 : 4;
  std::ostringstream ss;
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n";
  ss << "using scalar_t = " << scalar << ";\n";
  ss << "struct SdpaParams {\n";
  ss << "  uint B; uint H; uint Q; uint K; uint D; uint DV;\n";
  ss << "  uint has_mask; uint mask_B; uint mask_H; uint mask_Q; uint "
        "mask_K;\n";
  ss << "  uint scale_bits;\n";
  ss << "  uint k_gqa; uint k_heads; uint v_gqa; uint v_heads;\n";
  ss << "};\n";
  ss << "kernel void " << entry_point << "(\n";
  ss << "  device const scalar_t* q [[buffer(0)]],\n";
  ss << "  device const scalar_t* k [[buffer(1)]],\n";
  ss << "  device const scalar_t* v [[buffer(2)]],\n";
  if (has_mask) {
    ss << "  device const scalar_t* mask [[buffer(3)]],\n";
  }
  ss << "  constant SdpaParams& p [[buffer(" << params_buffer << ")]],\n";
  ss << "  device scalar_t* out [[buffer(" << output_buffer << ")]],\n";
  ss << "  uint gid [[thread_position_in_grid]],\n";
  ss << "  uint lane [[thread_index_in_threadgroup]],\n";
  ss << "  uint tgid [[threadgroup_position_in_grid]]) {\n";
  ss << "  if (p.D <= 64 && p.DV <= 64) {\n";
  ss << "    uint total_vectors = p.B * p.H * p.Q;\n";
  ss << "    if (tgid >= total_vectors) return;\n";
  ss << "    uint qi = tgid % p.Q;\n";
  ss << "    uint tmp_vec = tgid / p.Q;\n";
  ss << "    uint h = tmp_vec % p.H;\n";
  ss << "    uint b = tmp_vec / p.H;\n";
  ss << "    uint kh = (p.k_gqa != 0 && p.k_heads != 0) ? min(h / max(p.H / "
        "p.k_heads, 1u), p.k_heads - 1u) : h;\n";
  ss << "    uint vh = (p.v_gqa != 0 && p.v_heads != 0) ? min(h / max(p.H / "
        "p.v_heads, 1u), p.v_heads - 1u) : h;\n";
  ss << "    float scale = as_type<float>(p.scale_bits);\n";
  ss << "    float acc0 = 0.0f;\n";
  ss << "    float acc1 = 0.0f;\n";
  ss << "    float m = -INFINITY;\n";
  ss << "    float l = 0.0f;\n";
  ss << "    uint q_base = (((b * p.H + h) * p.Q + qi) * p.D);\n";
  ss << "    for (uint kk = 0; kk < p.K; ++kk) {\n";
  ss << "      float qk = 0.0f;\n";
  ss << "      uint k_base = (((b * (p.k_gqa != 0 ? p.k_heads : p.H) + kh) * "
        "p.K + kk) * p.D);\n";
  ss << "      for (uint d = lane; d < p.D; d += 32) {\n";
  ss << "        qk += float(q[q_base + d]) * float(k[k_base + d]);\n";
  ss << "      }\n";
  ss << "      float score = simd_sum(qk) * scale;\n";
  if (has_mask) {
    ss << "      if (p.has_mask != 0) {\n";
    ss << "        uint mb = (p.mask_B == 1) ? 0 : b;\n";
    ss << "        uint mh = (p.mask_H == 1) ? 0 : h;\n";
    ss << "        uint mq = (p.mask_Q == 1) ? 0 : qi;\n";
    ss << "        uint mk = (p.mask_K == 1) ? 0 : kk;\n";
    ss << "        uint mask_idx = (((mb * p.mask_H + mh) * p.mask_Q + mq) * "
          "p.mask_K + mk);\n";
    ss << "        score += float(mask[mask_idx]);\n";
    ss << "      }\n";
  }
  ss << "      float new_m = max(m, score);\n";
  ss << "      float old_scale = exp(m - new_m);\n";
  ss << "      float score_scale = exp(score - new_m);\n";
  ss << "      l = l * old_scale + score_scale;\n";
  ss << "      m = new_m;\n";
  ss << "      uint v_base = (((b * (p.v_gqa != 0 ? p.v_heads : p.H) + vh) * "
        "p.K + kk) * p.DV);\n";
  ss << "      if (lane < p.DV) {\n";
  ss << "        acc0 = acc0 * old_scale + score_scale * float(v[v_base + "
        "lane]);\n";
  ss << "      }\n";
  ss << "      if (lane + 32 < p.DV) {\n";
  ss << "        acc1 = acc1 * old_scale + score_scale * float(v[v_base + lane "
        "+ 32]);\n";
  ss << "      }\n";
  ss << "    }\n";
  ss << "    uint out_base = (((b * p.H + h) * p.Q + qi) * p.DV);\n";
  ss << "    if (lane < p.DV) {\n";
  ss << "      out[out_base + lane] = scalar_t(acc0 / l);\n";
  ss << "    }\n";
  ss << "    if (lane + 32 < p.DV) {\n";
  ss << "      out[out_base + lane + 32] = scalar_t(acc1 / l);\n";
  ss << "    }\n";
  ss << "    return;\n";
  ss << "  }\n";
  ss << "  uint total = p.B * p.H * p.Q * p.DV;\n";
  ss << "  if (gid >= total) return;\n";
  ss << "  uint dv = gid % p.DV;\n";
  ss << "  uint tmp = gid / p.DV;\n";
  ss << "  uint qi = tmp % p.Q;\n";
  ss << "  tmp /= p.Q;\n";
  ss << "  uint h = tmp % p.H;\n";
  ss << "  uint b = tmp / p.H;\n";
  ss << "  uint kh = (p.k_gqa != 0 && p.k_heads != 0) ? min(h / max(p.H / "
        "p.k_heads, 1u), p.k_heads - 1u) : h;\n";
  ss << "  uint vh = (p.v_gqa != 0 && p.v_heads != 0) ? min(h / max(p.H / "
        "p.v_heads, 1u), p.v_heads - 1u) : h;\n";
  ss << "  float scale = as_type<float>(p.scale_bits);\n";
  ss << "  float max_score = -INFINITY;\n";
  ss << "  for (uint kk = 0; kk < p.K; ++kk) {\n";
  ss << "    float score = 0.0f;\n";
  ss << "    uint q_base = (((b * p.H + h) * p.Q + qi) * p.D);\n";
  ss << "    uint k_base = (((b * (p.k_gqa != 0 ? p.k_heads : p.H) + kh) * p.K "
        "+ kk) * p.D);\n";
  ss << "    for (uint d = 0; d < p.D; ++d) {\n";
  ss << "      score += float(q[q_base + d]) * float(k[k_base + d]);\n";
  ss << "    }\n";
  ss << "    score *= scale;\n";
  if (has_mask) {
    ss << "    if (p.has_mask != 0) {\n";
    ss << "      uint mb = (p.mask_B == 1) ? 0 : b;\n";
    ss << "      uint mh = (p.mask_H == 1) ? 0 : h;\n";
    ss << "      uint mq = (p.mask_Q == 1) ? 0 : qi;\n";
    ss << "      uint mk = (p.mask_K == 1) ? 0 : kk;\n";
    ss << "      uint mask_idx = (((mb * p.mask_H + mh) * p.mask_Q + mq) * "
          "p.mask_K + mk);\n";
    ss << "      score += float(mask[mask_idx]);\n";
    ss << "    }\n";
  }
  ss << "    max_score = max(max_score, score);\n";
  ss << "  }\n";
  ss << "  float sum = 0.0f;\n";
  ss << "  float acc = 0.0f;\n";
  ss << "  for (uint kk = 0; kk < p.K; ++kk) {\n";
  ss << "    float score = 0.0f;\n";
  ss << "    uint q_base = (((b * p.H + h) * p.Q + qi) * p.D);\n";
  ss << "    uint k_base = (((b * (p.k_gqa != 0 ? p.k_heads : p.H) + kh) * p.K "
        "+ kk) * p.D);\n";
  ss << "    for (uint d = 0; d < p.D; ++d) {\n";
  ss << "      score += float(q[q_base + d]) * float(k[k_base + d]);\n";
  ss << "    }\n";
  ss << "    score *= scale;\n";
  if (has_mask) {
    ss << "    if (p.has_mask != 0) {\n";
    ss << "      uint mb = (p.mask_B == 1) ? 0 : b;\n";
    ss << "      uint mh = (p.mask_H == 1) ? 0 : h;\n";
    ss << "      uint mq = (p.mask_Q == 1) ? 0 : qi;\n";
    ss << "      uint mk = (p.mask_K == 1) ? 0 : kk;\n";
    ss << "      uint mask_idx = (((mb * p.mask_H + mh) * p.mask_Q + mq) * "
          "p.mask_K + mk);\n";
    ss << "      score += float(mask[mask_idx]);\n";
    ss << "    }\n";
  }
  ss << "    float w = exp(score - max_score);\n";
  ss << "    sum += w;\n";
  ss << "    uint v_idx = (((b * (p.v_gqa != 0 ? p.v_heads : p.H) + vh) * p.K "
        "+ kk) * p.DV + dv);\n";
  ss << "    acc += w * float(v[v_idx]);\n";
  ss << "  }\n";
  ss << "  out[gid] = scalar_t(acc / sum);\n";
  ss << "}\n";
  return ss.str();
}

} // namespace

std::string generate_msl_for_sdpa(ov::element::Type type) {
  return generate_msl_for_sdpa_variant(type, /*has_mask=*/true);
}

std::string generate_msl_for_sdpa_with_causal_mask(ov::element::Type type) {
  const std::string scalar = metal_scalar_type(type);
  std::ostringstream ss;
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n";
  ss << "using scalar_t = " << scalar << ";\n";
  ss << "struct SdpaCausalMaskParams {\n";
  ss << "  uint B; uint H; uint Q; uint K; uint D; uint DV;\n";
  ss << "  uint mask_K; uint scale_bits; uint k_gqa; uint k_heads; uint v_gqa; "
        "uint v_heads;\n";
  ss << "};\n";
  ss << "inline float gfx_sdpa_mask_score(device const long* attention_mask,\n";
  ss << "                                  device const long* "
        "cache_positions,\n";
  ss << "                                  constant SdpaCausalMaskParams& p,\n";
  ss << "                                  uint b, uint qi, uint kk) {\n";
  ss << "  long row = cache_positions[qi];\n";
  ss << "  bool causal_block = long(kk) > row;\n";
  ss << "  bool padding_block = false;\n";
  ss << "  if (kk < p.mask_K) {\n";
  ss << "    padding_block = attention_mask[b * p.mask_K + kk] == 0;\n";
  ss << "  }\n";
  ss << "  return (causal_block || padding_block) ? -INFINITY : 0.0f;\n";
  ss << "}\n";
  ss << "kernel void sdpa_causal_mask_kernel(\n";
  ss << "  device const scalar_t* q [[buffer(0)]],\n";
  ss << "  device const scalar_t* k [[buffer(1)]],\n";
  ss << "  device const scalar_t* v [[buffer(2)]],\n";
  ss << "  device const long* attention_mask [[buffer(3)]],\n";
  ss << "  device const long* cache_positions [[buffer(4)]],\n";
  ss << "  constant SdpaCausalMaskParams& p [[buffer(5)]],\n";
  ss << "  device scalar_t* out [[buffer(6)]],\n";
  ss << "  uint gid [[thread_position_in_grid]],\n";
  ss << "  uint lane [[thread_index_in_threadgroup]],\n";
  ss << "  uint tgid [[threadgroup_position_in_grid]]) {\n";
  ss << "  if (p.D <= 64 && p.DV <= 64) {\n";
  ss << "    uint total_vectors = p.B * p.H * p.Q;\n";
  ss << "    if (tgid >= total_vectors) return;\n";
  ss << "    uint qi = tgid % p.Q;\n";
  ss << "    uint tmp_vec = tgid / p.Q;\n";
  ss << "    uint h = tmp_vec % p.H;\n";
  ss << "    uint b = tmp_vec / p.H;\n";
  ss << "    uint kh = (p.k_gqa != 0 && p.k_heads != 0) ? min(h / max(p.H / "
        "p.k_heads, 1u), p.k_heads - 1u) : h;\n";
  ss << "    uint vh = (p.v_gqa != 0 && p.v_heads != 0) ? min(h / max(p.H / "
        "p.v_heads, 1u), p.v_heads - 1u) : h;\n";
  ss << "    float scale = as_type<float>(p.scale_bits);\n";
  ss << "    float acc0 = 0.0f;\n";
  ss << "    float acc1 = 0.0f;\n";
  ss << "    float m = -INFINITY;\n";
  ss << "    float l = 0.0f;\n";
  ss << "    uint q_base = (((b * p.H + h) * p.Q + qi) * p.D);\n";
  ss << "    for (uint kk = 0; kk < p.K; ++kk) {\n";
  ss << "      float qk = 0.0f;\n";
  ss << "      uint k_base = (((b * (p.k_gqa != 0 ? p.k_heads : p.H) + kh) * "
        "p.K + kk) * p.D);\n";
  ss << "      for (uint d = lane; d < p.D; d += 32) {\n";
  ss << "        qk += float(q[q_base + d]) * float(k[k_base + d]);\n";
  ss << "      }\n";
  ss << "      float score = simd_sum(qk) * scale + "
        "gfx_sdpa_mask_score(attention_mask, cache_positions, p, b, qi, kk);\n";
  ss << "      float new_m = max(m, score);\n";
  ss << "      float old_scale = exp(m - new_m);\n";
  ss << "      float score_scale = exp(score - new_m);\n";
  ss << "      l = l * old_scale + score_scale;\n";
  ss << "      m = new_m;\n";
  ss << "      uint v_base = (((b * (p.v_gqa != 0 ? p.v_heads : p.H) + vh) * "
        "p.K + kk) * p.DV);\n";
  ss << "      if (lane < p.DV) acc0 = acc0 * old_scale + score_scale * "
        "float(v[v_base + lane]);\n";
  ss << "      if (lane + 32 < p.DV) acc1 = acc1 * old_scale + score_scale * "
        "float(v[v_base + lane + 32]);\n";
  ss << "    }\n";
  ss << "    uint out_base = (((b * p.H + h) * p.Q + qi) * p.DV);\n";
  ss << "    if (lane < p.DV) out[out_base + lane] = scalar_t(acc0 / l);\n";
  ss << "    if (lane + 32 < p.DV) out[out_base + lane + 32] = scalar_t(acc1 / "
        "l);\n";
  ss << "    return;\n";
  ss << "  }\n";
  ss << "  uint total = p.B * p.H * p.Q * p.DV;\n";
  ss << "  if (gid >= total) return;\n";
  ss << "  uint dv = gid % p.DV;\n";
  ss << "  uint tmp = gid / p.DV;\n";
  ss << "  uint qi = tmp % p.Q;\n";
  ss << "  tmp /= p.Q;\n";
  ss << "  uint h = tmp % p.H;\n";
  ss << "  uint b = tmp / p.H;\n";
  ss << "  uint kh = (p.k_gqa != 0 && p.k_heads != 0) ? min(h / max(p.H / "
        "p.k_heads, 1u), p.k_heads - 1u) : h;\n";
  ss << "  uint vh = (p.v_gqa != 0 && p.v_heads != 0) ? min(h / max(p.H / "
        "p.v_heads, 1u), p.v_heads - 1u) : h;\n";
  ss << "  float scale = as_type<float>(p.scale_bits);\n";
  ss << "  float max_score = -INFINITY;\n";
  ss << "  for (uint kk = 0; kk < p.K; ++kk) {\n";
  ss << "    float score = 0.0f;\n";
  ss << "    uint q_base = (((b * p.H + h) * p.Q + qi) * p.D);\n";
  ss << "    uint k_base = (((b * (p.k_gqa != 0 ? p.k_heads : p.H) + kh) * p.K "
        "+ kk) * p.D);\n";
  ss << "    for (uint d = 0; d < p.D; ++d) score += float(q[q_base + d]) * "
        "float(k[k_base + d]);\n";
  ss << "    score = score * scale + gfx_sdpa_mask_score(attention_mask, "
        "cache_positions, p, b, qi, kk);\n";
  ss << "    max_score = max(max_score, score);\n";
  ss << "  }\n";
  ss << "  float sum = 0.0f;\n";
  ss << "  float acc = 0.0f;\n";
  ss << "  for (uint kk = 0; kk < p.K; ++kk) {\n";
  ss << "    float score = 0.0f;\n";
  ss << "    uint q_base = (((b * p.H + h) * p.Q + qi) * p.D);\n";
  ss << "    uint k_base = (((b * (p.k_gqa != 0 ? p.k_heads : p.H) + kh) * p.K "
        "+ kk) * p.D);\n";
  ss << "    for (uint d = 0; d < p.D; ++d) score += float(q[q_base + d]) * "
        "float(k[k_base + d]);\n";
  ss << "    score = score * scale + gfx_sdpa_mask_score(attention_mask, "
        "cache_positions, p, b, qi, kk);\n";
  ss << "    float w = exp(score - max_score);\n";
  ss << "    sum += w;\n";
  ss << "    uint v_idx = (((b * (p.v_gqa != 0 ? p.v_heads : p.H) + vh) * p.K "
        "+ kk) * p.DV + dv);\n";
  ss << "    acc += w * float(v[v_idx]);\n";
  ss << "  }\n";
  ss << "  out[gid] = scalar_t(acc / sum);\n";
  ss << "}\n";
  return ss.str();
}

GfxKernelRuntimeBindingPlan make_causal_sdpa_backend_binding_plan() {
  return make_backend_custom_kernel_binding_plan(
      /*is_opencl_backend=*/false, "GfxSDPAWithCausalMask",
      "sdpa_causal_mask_kernel");
}

GfxKernelRuntimeBindingPlan make_sdpa_backend_binding_plan(bool has_mask) {
  return make_backend_custom_kernel_binding_plan(
      /*is_opencl_backend=*/false, "ScaledDotProductAttention",
      has_mask ? "sdpa_kernel" : "sdpa_nomask_kernel");
}

GfxMslGeneratedKernelSourcePlan
make_sdpa_msl_kernel_source_plan(ov::element::Type type, bool has_mask) {
  KernelSource source;
  source.entry_point = has_mask ? "sdpa_kernel" : "sdpa_nomask_kernel";
  source.msl_source = generate_msl_for_sdpa_variant(type, has_mask);
  return make_msl_generated_custom_kernel_source_plan(
      std::move(source), "ScaledDotProductAttention");
}

GfxMslGeneratedKernelSourcePlan
make_causal_sdpa_msl_kernel_source_plan(ov::element::Type type) {
  KernelSource source;
  source.entry_point = "sdpa_causal_mask_kernel";
  source.msl_source = generate_msl_for_sdpa_with_causal_mask(type);
  return make_msl_generated_custom_kernel_source_plan(
      std::move(source), "GfxSDPAWithCausalMask");
}

GfxSdpaMslRuntimeParamsPlan make_causal_sdpa_msl_runtime_params_plan(
    const ov::Shape &q_shape, const ov::Shape &k_shape,
    const ov::Shape &v_shape, const ov::Shape &mask_shape, float scale,
    bool k_gqa, size_t k_heads, bool v_gqa, size_t v_heads) {
  GfxSdpaMslRuntimeParamsPlan plan{};
  plan.params = {
      static_cast<int32_t>(q_shape[0]),
      static_cast<int32_t>(q_shape[1]),
      static_cast<int32_t>(q_shape[2]),
      static_cast<int32_t>(k_shape[2]),
      static_cast<int32_t>(q_shape[3]),
      static_cast<int32_t>(v_shape[3]),
      static_cast<int32_t>(mask_shape[1]),
      bitcast_f32_to_i32(scale),
      k_gqa ? 1 : 0,
      static_cast<int32_t>(k_heads),
      v_gqa ? 1 : 0,
      static_cast<int32_t>(v_heads),
      0,
      0,
      0,
      0,
  };
  plan.binding = make_causal_sdpa_backend_binding_plan();
  return plan;
}

GfxSdpaMslRuntimeParamsPlan make_sdpa_msl_runtime_params_plan(
    const ov::Shape &q_shape, const ov::Shape &k_shape,
    const ov::Shape &v_shape, const ov::Shape &mask_shape, bool has_mask,
    float scale, bool k_gqa, size_t k_heads, bool v_gqa, size_t v_heads) {
  GfxSdpaMslRuntimeParamsPlan plan{};
  plan.params = {
      static_cast<int32_t>(q_shape[0]),
      static_cast<int32_t>(q_shape[1]),
      static_cast<int32_t>(q_shape[2]),
      static_cast<int32_t>(k_shape[2]),
      static_cast<int32_t>(q_shape[3]),
      static_cast<int32_t>(v_shape[3]),
      has_mask ? 1 : 0,
      static_cast<int32_t>(mask_shape[0]),
      static_cast<int32_t>(mask_shape[1]),
      static_cast<int32_t>(mask_shape[2]),
      static_cast<int32_t>(mask_shape[3]),
      bitcast_f32_to_i32(scale),
      k_gqa ? 1 : 0,
      static_cast<int32_t>(k_heads),
      v_gqa ? 1 : 0,
      static_cast<int32_t>(v_heads),
  };
  plan.binding = make_sdpa_backend_binding_plan(has_mask);
  return plan;
}

} // namespace gfx_plugin
} // namespace ov
