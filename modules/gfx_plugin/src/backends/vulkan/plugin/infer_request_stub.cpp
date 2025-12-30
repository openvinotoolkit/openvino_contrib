// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include "openvino/gfx_plugin/compiled_model.hpp"
#include "openvino/core/except.hpp"
#include "plugin/infer_request_state.hpp"

namespace ov {
namespace gfx_plugin {

void VulkanProfilerDeleter::operator()(VulkanProfiler* ptr) const {
    (void)ptr;
}

void InferRequest::infer_vulkan_impl(const std::shared_ptr<const CompiledModel>& cm) {
    OPENVINO_ASSERT(cm, "CompiledModel is null");
    OPENVINO_THROW("GFX Vulkan backend is not available in this build");
}

void InferRequest::release_vulkan_cache() {}

}  // namespace gfx_plugin
}  // namespace ov
