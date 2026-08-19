// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_target_profile.hpp"

#include "runtime/gfx_profiler.hpp"

namespace ov {
namespace gfx_plugin {

GfxTargetProfile make_gfx_target_profile(const GpuExecutionDeviceInfo& info) {
    GfxTargetProfile profile;
    profile.backend = info.backend;
    profile.device_family = info.device_family;
    profile.device_key = info.device_key;
    profile.device_name = info.device_name;
    profile.vendor_id = info.vendor_id;
    profile.device_id = info.device_id;
    profile.driver_version = info.driver_version;
    profile.api_version = info.api_version;
    profile.preferred_simd_width = info.preferred_simd_width;
    profile.subgroup_size = info.subgroup_size;
    profile.max_total_threads_per_group = info.max_total_threads_per_group;
    profile.max_threads_per_group = info.max_threads_per_group;
    profile.min_storage_buffer_offset_alignment = info.min_storage_buffer_offset_alignment;
    profile.non_coherent_atom_size = info.non_coherent_atom_size;
    profile.supports_storage_buffer_8bit = info.supports_storage_buffer_8bit;
    profile.supports_storage_buffer_16bit = info.supports_storage_buffer_16bit;
    profile.supports_shader_float16 = info.supports_shader_float16;
    profile.supports_shader_int8 = info.supports_shader_int8;
    profile.supports_conv_output_channel_blocking = info.supports_conv_output_channel_blocking;
    profile.supports_conv_channel_block_spatial_tiling = info.supports_conv_channel_block_spatial_tiling;
    return profile;
}

void record_gfx_target_profile(const GfxTargetProfile& profile, GfxProfiler* profiler) {
    if (!profiler) {
        return;
    }
    profiler->set_target_profile(profile);
    profiler->set_counter("target_backend_metal", profile.backend == GpuBackend::Metal ? 1 : 0);
    profiler->set_counter("target_backend_opencl", profile.backend == GpuBackend::OpenCL ? 1 : 0);
    profiler->set_counter("target_preferred_simd_width", profile.preferred_simd_width);
    profiler->set_counter("target_subgroup_size", profile.subgroup_size);
    profiler->set_counter("target_max_total_threads_per_group", profile.max_total_threads_per_group);
    profiler->set_counter("target_max_threads_per_group_x", profile.max_threads_per_group[0]);
    profiler->set_counter("target_max_threads_per_group_y", profile.max_threads_per_group[1]);
    profiler->set_counter("target_max_threads_per_group_z", profile.max_threads_per_group[2]);
    profiler->set_counter("target_supports_storage_buffer_8bit", profile.supports_storage_buffer_8bit ? 1 : 0);
    profiler->set_counter("target_supports_storage_buffer_16bit", profile.supports_storage_buffer_16bit ? 1 : 0);
    profiler->set_counter("target_supports_shader_float16", profile.supports_shader_float16 ? 1 : 0);
    profiler->set_counter("target_supports_shader_int8", profile.supports_shader_int8 ? 1 : 0);
    profiler->set_counter("target_supports_conv_output_channel_blocking",
                          profile.supports_conv_output_channel_blocking ? 1 : 0);
    profiler->set_counter("target_supports_conv_channel_block_spatial_tiling",
                          profile.supports_conv_channel_block_spatial_tiling ? 1 : 0);
}

}  // namespace gfx_plugin
}  // namespace ov
