// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <limits>

namespace ov {
namespace gfx_plugin {

struct RuntimeOutputLifetime {
  static constexpr size_t npos = std::numeric_limits<size_t>::max();

  size_t produced_at = npos;
  size_t last_used_at = npos;
  bool requires_buffer = true;
  size_t storage_source_output = npos;

  bool valid() const noexcept {
    return produced_at != npos && last_used_at != npos;
  }
};

}  // namespace gfx_plugin
}  // namespace ov
