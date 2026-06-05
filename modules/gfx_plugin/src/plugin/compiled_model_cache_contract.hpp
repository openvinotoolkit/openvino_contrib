// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

[[noreturn]] inline void
throw_compiled_model_cache_roundtrip_unavailable(const char *api_name) {
  OPENVINO_THROW("GFX: ", api_name,
                 " requires a serialized CacheEnvelope/ExecutableBundle. "
                 "The old OpenVINO-model serialization path is disabled.");
}

} // namespace gfx_plugin
} // namespace ov
