// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/softmax_f16_dynamic_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f32_dynamic_kernel.hpp"
#include "kernel_ir/opencl_kernels/softmax_f32_kernel.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<GfxOpenClSourceArtifact> make_opencl_softmax_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view requested_kernel_unit_id = {});

} // namespace gfx_plugin
} // namespace ov
