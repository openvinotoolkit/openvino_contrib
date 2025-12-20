// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/core/type/element_type.hpp"

#include "runtime/metal_memory.hpp"
#include "runtime/metal_op.hpp"
#include "runtime/metal_op_factory.hpp"
#include "runtime/profiling/metal_profiler_config.hpp"
#include "runtime/profiling/metal_profiling_report.hpp"

#include "plugin/metal_properties.hpp"

namespace ov {
namespace metal_plugin {

class Plugin;
class InferRequest;
struct OutputDesc {
    ov::Shape shape;
    ov::element::Type type = ov::element::dynamic;
    bool is_model_output = false;
};

struct PipelineStageDesc {
    std::shared_ptr<const ov::Node> node;
    std::unique_ptr<MetalOp> op;  // compiled prototype
    struct InputLink {
        std::shared_ptr<const ov::Node> node;
        size_t port = 0;
    };
    std::vector<OutputDesc> outputs;
    std::vector<InputLink> inputs;
};

class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const std::shared_ptr<const ov::Model>& original_model = nullptr,
                  const ov::AnyMap& properties = {},
                  const ov::SoPtr<ov::IRemoteContext>& context = {});
    ~CompiledModel() override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override { return m_runtime_model; }
    void export_model(std::ostream& model) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    ov::element::Type get_inference_precision() const { return m_inference_precision; }
    std::shared_ptr<MetalBufferManager> const_manager() const { return m_const_manager; }
    MetalAllocatorCore& allocator_core() const { return *m_alloc_core; }
    const MetalDeviceCaps& device_caps() const { return m_caps; }
    MetalDeviceHandle device_handle() const { return m_device; }
    MetalCommandQueueHandle command_queue() const { return m_command_queue; }
    const MetalMemoryStats& memory_stats() const { return m_dummy_stats; }
    void update_last_stats(const MetalMemoryStats& stats) const { m_last_stats = stats; }
    bool enable_profiling() const { return m_enable_profiling; }
    ProfilingLevel profiling_level() const;
    size_t op_pipeline_size() const { return m_pipeline.size(); }
    bool op_pipeline_built() const { return m_pipeline_built; }
    const std::vector<PipelineStageDesc>& pipeline_desc() const { return m_pipeline; }
    const std::unordered_map<const ov::Node*, size_t>& node_to_stage() const { return m_node_to_stage; }
    const std::unordered_map<const ov::Node*, size_t>& parameter_index() const { return m_param_index; }
    void update_last_profiling_report(const MetalProfilingReport& report) const;
    std::string last_profiling_report_json() const;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    void build_op_pipeline();

    MetalDeviceHandle m_device = nullptr;
    MetalCommandQueueHandle m_command_queue = nullptr;
    MetalDeviceCaps m_caps{};
    std::unique_ptr<MetalAllocatorCore> m_alloc_core;
    std::unique_ptr<MetalHeapPool> m_persistent_heaps;
    std::unique_ptr<MetalFreeList> m_persistent_freelist;
    std::unique_ptr<MetalStagingPool> m_persistent_staging;
    std::unique_ptr<MetalAllocator> m_persistent_alloc;
    std::unique_ptr<MetalConstCache> m_const_cache;
    std::shared_ptr<MetalBufferManager> m_const_manager;
    std::shared_ptr<const ov::Model> m_runtime_model;
    std::shared_ptr<const ov::Model> m_original_model;
    ov::AnyMap m_config;
    ov::element::Type m_inference_precision{ov::element::f32};
    bool m_enable_profiling = false;
    ProfilingLevel m_profiling_level = ProfilingLevel::Standard;
    bool m_profiling_level_set = false;
    bool m_pipeline_built = false;
    mutable std::vector<PipelineStageDesc> m_pipeline;
    std::unordered_map<const ov::Node*, size_t> m_node_to_stage;
    std::unordered_map<const ov::Node*, size_t> m_param_index;
    MetalMemoryStats m_dummy_stats{};
    mutable MetalMemoryStats m_last_stats{};
    mutable std::mutex m_report_mutex;
    mutable std::string m_last_report_json;
};

}  // namespace metal_plugin
}  // namespace ov
