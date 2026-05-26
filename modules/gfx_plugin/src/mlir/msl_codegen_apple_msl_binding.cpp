// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <utility>

#include "mlir/gfx_backend_custom_kernel_adapter.hpp"

namespace ov {
namespace gfx_plugin {

void require_apple_msl_generated_kernel_source_binding(
    KernelSource &source, std::string_view stage_type,
    std::string_view entry_point, std::vector<int32_t> scalar_args) {
  require_backend_custom_kernel_source_binding(
      source, /*is_opencl_backend=*/false, stage_type, entry_point,
      std::move(scalar_args));
}

} // namespace gfx_plugin
} // namespace ov
