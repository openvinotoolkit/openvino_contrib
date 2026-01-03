// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace gfx_plugin {

// Plugin-level properties.
constexpr const char* kGfxProfilingLevelProperty = "GFX_PROFILING_LEVEL";
constexpr const char* kGfxProfilingReportProperty = "GFX_PROFILING_REPORT";
constexpr const char* kGfxMemStatsProperty = "GFX_MEM_STATS";
constexpr const char* kGfxBackendProperty = "GFX_BACKEND";
constexpr const char* kGfxEnableFusionProperty = "GFX_ENABLE_FUSION";

// Common remote tensor/property keys.
constexpr const char* kGfxBufferProperty = "GFX_BUFFER";
constexpr const char* kGfxMemoryProperty = "GFX_MEMORY";
constexpr const char* kGfxBufferBytesProperty = "GFX_BUFFER_BYTES";
constexpr const char* kGfxHostVisibleProperty = "GFX_HOST_VISIBLE";
constexpr const char* kGfxStorageModeProperty = "GFX_STORAGE_MODE";

}  // namespace gfx_plugin
}  // namespace ov
