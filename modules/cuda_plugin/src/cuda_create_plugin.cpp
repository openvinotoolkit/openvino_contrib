// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_plugin.hpp"

namespace CUDAPlugin {
namespace {

const InferenceEngine::Version version{{2, 1}, CI_BUILD_NUMBER, "openvino_cuda_plugin"};

IE_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace
}  // namespace CUDAPlugin
