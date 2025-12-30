// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include "openvino/core/except.hpp"
#include "openvino/gfx_plugin/compiled_model.hpp"
#include "plugin/infer_request_backend_hooks.hpp"
#include "plugin/infer_request_state.hpp"

namespace ov {
namespace gfx_plugin {

void MetalInferStateDeleter::operator()(MetalInferState* ptr) const {
    (void)ptr;
}

void init_backend_infer_state(InferRequestState& /*state*/, const CompiledModel& /*cm*/) {}

ov::SoPtr<ov::ITensor> get_backend_tensor_override(const InferRequestState& /*state*/,
                                                   size_t /*idx*/,
                                                   const std::vector<ov::Output<const ov::Node>>& /*outputs*/) {
    return {};
}

void InferRequest::infer_metal_impl(const std::shared_ptr<const CompiledModel>& /*cm*/) {
    OPENVINO_THROW("GFX Metal backend is not available in this build");
}

}  // namespace gfx_plugin
}  // namespace ov
