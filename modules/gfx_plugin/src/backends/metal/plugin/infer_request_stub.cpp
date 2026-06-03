// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include "openvino/core/except.hpp"
#include "backends/metal/plugin/compiled_model_state.hpp"
namespace ov {
namespace gfx_plugin {

void MetalBackendState::init_infer_state(BackendRequestState& /*state*/) const {}

ov::SoPtr<ov::ITensor> MetalBackendState::get_tensor_override(
    const BackendRequestState& /*state*/,
    size_t /*idx*/,
    const std::vector<ov::Output<const ov::Node>>& /*outputs*/) const {
    return {};
}

void execute_metal_infer_request(InferRequest&,
                                 const std::shared_ptr<const CompiledModel>& /*cm*/) {
    OPENVINO_THROW("GFX Metal backend is not available in this build");
}

}  // namespace gfx_plugin
}  // namespace ov
