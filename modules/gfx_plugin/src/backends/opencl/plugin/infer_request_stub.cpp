// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/gfx_plugin/infer_request.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

void InferRequest::infer_opencl_impl(const std::shared_ptr<const CompiledModel>&) {
    OPENVINO_THROW("GFX OpenCL backend is not available in this build");
}

}  // namespace gfx_plugin
}  // namespace ov
