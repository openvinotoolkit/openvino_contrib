// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/runtime.hpp>
#include "cuda_plugin.hpp"

namespace CUDAPlugin {
template <>
std::string Plugin::getCudaAttribute<Plugin::cuda_attribute::name, std::string>() const {
  return CUDA::Device{cudaDeviceID()}.props().name;
}
} // namespace CUDAPlugin
