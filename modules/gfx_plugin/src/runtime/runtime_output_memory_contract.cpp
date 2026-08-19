// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/runtime_output_memory_contract.hpp"

#include "runtime/infer_pipeline.hpp"

#include <algorithm>
#include <string_view>

namespace ov {
namespace gfx_plugin {

namespace {

bool same_runtime_region_kind(const RuntimeMemoryRegionDescriptor& region,
                              std::string_view kind) {
    return region.kind.size() == kind.size() && region.kind.compare(kind) == 0;
}

BufferUsage usage_for_runtime_memory_region(const RuntimeMemoryRegionDescriptor& region,
                                            const char* error_prefix) {
    if (same_runtime_region_kind(region, "external_tensor")) {
        return BufferUsage::IO;
    }
    if (same_runtime_region_kind(region, "immutable_tensor")) {
        return BufferUsage::Const;
    }
    if (same_runtime_region_kind(region, "transient_tensor")) {
        return region.host_visible ? BufferUsage::IO : BufferUsage::Intermediate;
    }
    OPENVINO_THROW(error_prefix,
                   ": unsupported runtime memory region kind '",
                   region.kind,
                   "' for region ",
                   region.region_id);
}

bool region_is_in_transient_arena(const RuntimeMemoryPlanDescriptor& plan,
                                  const RuntimeMemoryRegionDescriptor& region) {
    if (!same_runtime_region_kind(region, "transient_tensor") ||
        region.external_binding || region.host_visible) {
        return false;
    }
    for (const auto& arena : plan.transient_arenas) {
        if (arena.storage_kind != region.storage_kind) {
            continue;
        }
        if (std::find(arena.region_ids.begin(),
                      arena.region_ids.end(),
                      region.region_id) != arena.region_ids.end()) {
            return true;
        }
    }
    return false;
}

}  // namespace

bool runtime_output_uses_transient_arena(const InferStage& stage,
                                         size_t output_index) {
    if (!stage.runtime_session) {
        return false;
    }
    const auto* region = runtime_output_memory_region_or_null(stage, output_index);
    if (!region) {
        return false;
    }
    return region_is_in_transient_arena(stage.runtime_session->descriptor().memory_plan,
                                        *region);
}

bool apply_runtime_output_memory_contract(const InferStage& stage,
                                          size_t output_index,
                                          GpuBufferDesc& desc,
                                          GpuTensor& output,
                                          const char* error_prefix) {
    const auto* descriptor = runtime_stage_descriptor_or_null(stage);
    if (!descriptor) {
        return false;
    }
    OPENVINO_ASSERT(stage.runtime_session,
                    error_prefix,
                    ": runtime memory plan is required for stage ",
                    descriptor->stage_name);
    const auto& memory_plan = stage.runtime_session->descriptor().memory_plan;
    OPENVINO_ASSERT(!memory_plan.hidden_host_copies_allowed,
                    error_prefix,
                    ": runtime memory plan allows hidden host copies");
    OPENVINO_ASSERT(output_index < descriptor->output_bindings.size(),
                    error_prefix,
                    ": runtime output binding missing for stage ",
                    descriptor->stage_name,
                    " output ",
                    output_index);
    const auto& binding = descriptor->output_bindings[output_index];
    OPENVINO_ASSERT(!binding.memory_region_id.empty(),
                    error_prefix,
                    ": runtime output binding has no memory region for stage ",
                    descriptor->stage_name,
                    " output ",
                    output_index);
    const auto* region = runtime_output_memory_region_or_null(stage, output_index);
    OPENVINO_ASSERT(region,
                    error_prefix,
                    ": runtime memory region '",
                    binding.memory_region_id,
                    "' is missing for stage ",
                    descriptor->stage_name,
                    " output ",
                    output_index);
    OPENVINO_ASSERT(region->alias_group == binding.alias_group,
                    error_prefix,
                    ": runtime output binding alias group drifts from memory plan for stage ",
                    descriptor->stage_name,
                    " output ",
                    output_index);
    OPENVINO_ASSERT(region->storage_kind == binding.storage_kind,
                    error_prefix,
                    ": runtime output binding storage kind drifts from memory plan for stage ",
                    descriptor->stage_name,
                    " output ",
                    output_index);

    desc.usage = usage_for_runtime_memory_region(*region, error_prefix);
    if (desc.usage == BufferUsage::Const) {
        desc.cpu_read = false;
        desc.cpu_write = region->host_visible;
        desc.prefer_device_local = !region->host_visible;
    } else {
        const bool host_visible =
            region->host_visible || desc.usage == BufferUsage::IO;
        desc.cpu_read = host_visible;
        desc.cpu_write = host_visible;
        desc.prefer_device_local = !host_visible;
    }
    output.prefer_private = desc.prefer_device_local;
    return true;
}

}  // namespace gfx_plugin
}  // namespace ov
