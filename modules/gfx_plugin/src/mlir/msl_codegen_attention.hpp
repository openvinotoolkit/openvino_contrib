// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/msl_codegen_apple_msl_dispatch.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {

std::string generate_msl_for_sdpa(ov::element::Type type);
std::string generate_msl_for_sdpa_with_causal_mask(ov::element::Type type);

GfxMslGeneratedKernelSourcePlan
make_sdpa_msl_kernel_source_plan(ov::element::Type type, bool has_mask);
GfxMslGeneratedKernelSourcePlan
make_causal_sdpa_msl_kernel_source_plan(ov::element::Type type);

struct GfxSdpaMslRuntimeParamsPlan {
  std::vector<int32_t> params;
  GfxKernelRuntimeBindingPlan binding;

  bool valid() const { return !params.empty() && binding.valid; }
};

GfxSdpaMslRuntimeParamsPlan make_causal_sdpa_msl_runtime_params_plan(
    const ov::Shape &q_shape, const ov::Shape &k_shape,
    const ov::Shape &v_shape, const ov::Shape &mask_shape, float scale,
    bool k_gqa, size_t k_heads, bool v_gqa, size_t v_heads);
GfxSdpaMslRuntimeParamsPlan make_sdpa_msl_runtime_params_plan(
    const ov::Shape &q_shape, const ov::Shape &k_shape,
    const ov::Shape &v_shape, const ov::Shape &mask_shape, bool has_mask,
    float scale, bool k_gqa, size_t k_heads, bool v_gqa, size_t v_heads);

} // namespace gfx_plugin
} // namespace ov
