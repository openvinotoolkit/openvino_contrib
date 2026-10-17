// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

namespace ov {
namespace gfx_plugin {

struct GpuStageSubmitPolicy {
  size_t weight = 1;
  bool isolate = false;
};

} // namespace gfx_plugin
} // namespace ov
