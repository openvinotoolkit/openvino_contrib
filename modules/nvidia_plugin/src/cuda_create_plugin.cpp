// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_plugin.hpp"

namespace ov {
namespace nvidia_gpu {
namespace {

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_nvidia_gpu_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(Plugin, version)

}  // namespace
}  // namespace nvidia_gpu
}  // namespace ov
