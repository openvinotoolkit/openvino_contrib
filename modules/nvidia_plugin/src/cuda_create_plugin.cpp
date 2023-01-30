// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_plugin.hpp"

namespace ov {
namespace nvidia_gpu {
namespace {

const InferenceEngine::Version version{{2, 1}, CI_BUILD_NUMBER, "openvino_nvidia_gpu_plugin"};

IE_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace
}  // namespace nvidia_gpu
}  // namespace ov
