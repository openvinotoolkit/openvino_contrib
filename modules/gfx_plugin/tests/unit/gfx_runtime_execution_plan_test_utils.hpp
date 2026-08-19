// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/artifact_payload.hpp"
#include "openvino/core/except.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/runtime_execution_plan.hpp"

namespace ov {
namespace gfx_plugin {
namespace test {

class KernelPayload final : public KernelArtifactPayload {
public:
    explicit KernelPayload(std::string source_id)
        : m_source_id(std::move(source_id)) {}

    KernelArtifactPayloadKind payload_kind() const noexcept override {
        return KernelArtifactPayloadKind::MslSource;
    }

    std::string_view backend_domain() const noexcept override {
        return kBackendMetal;
    }

    std::string_view source_id() const noexcept override {
        return m_source_id;
    }

    std::string_view entry_point() const noexcept override {
        return m_source_id;
    }

    bool valid() const noexcept override {
        return !m_source_id.empty();
    }

private:
    std::string m_source_id;
};

namespace detail {

inline size_t
runtime_stage_slot_count(const std::vector<PipelineStageDesc>& descs) {
    size_t count = 0;
    for (const auto& desc : descs) {
        if (desc.runtime_stage_index != PipelineStageDesc::npos) {
            count = std::max(count, desc.runtime_stage_index + 1);
        }
    }
    return count;
}

inline std::string shape_contract(const ov::Shape& shape) {
    std::string result = "{";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            result += ",";
        }
        result += std::to_string(shape[i]);
    }
    result += "}";
    return result;
}

class RuntimePlanStageFactory final : public BackendStageFactory {
public:
    explicit RuntimePlanStageFactory(std::vector<PipelineStageDesc> descs)
        : m_stages(runtime_stage_slot_count(descs)) {
        for (auto& desc : descs) {
            OPENVINO_ASSERT(desc.stage,
                            "GFX unit test: runtime plan stage is null");
            OPENVINO_ASSERT(
                desc.runtime_stage_index != PipelineStageDesc::npos &&
                    desc.runtime_stage_index < m_stages.size(),
                "GFX unit test: runtime plan stage index is invalid");
            m_stages[desc.runtime_stage_index] = std::move(desc.stage);
        }
    }

    GpuBackend backend() const override {
        return GpuBackend::Metal;
    }

    std::unique_ptr<GpuStage> create_stage(
        const RuntimeStageMaterializationContext& context) const override {
        const auto index = context.require_descriptor().stage_index;
        OPENVINO_ASSERT(index < m_stages.size() && m_stages[index],
                        "GFX unit test: runtime plan stage ", index,
                        " was already materialized or is missing");
        return std::move(m_stages[index]);
    }

private:
    mutable std::vector<std::unique_ptr<GpuStage>> m_stages;
};

inline void complete_stage_descriptor(RuntimeStageExecutableDescriptor& stage,
                                      const PipelineStageDesc& desc) {
    stage.origin = KernelArtifactOrigin::Generated;
    stage.payload_kind = KernelArtifactPayloadKind::MslSource;
    stage.entry_point = "test_entry_" + std::to_string(stage.stage_index);
    OPENVINO_ASSERT(stage.output_bindings.size() == desc.outputs.size(),
                    "GFX unit test: runtime output binding count drift");
    for (size_t i = 0; i < desc.outputs.size(); ++i) {
        const auto& output = desc.outputs[i];
        auto& binding = stage.output_bindings[i];
        if (output.type != ov::element::dynamic) {
            binding.element_type = output.type.get_type_name();
        }
        if (!output.shape.empty()) {
            binding.partial_shape = shape_contract(output.shape);
        }
    }
    stage.abi_arg_count =
        static_cast<uint32_t>(stage.input_bindings.size());
    stage.abi_output_arg_count =
        static_cast<uint32_t>(stage.output_bindings.size());
    stage.payload = std::make_shared<KernelPayload>(stage.entry_point);
}

inline PipelineStageMaterializationPlan make_materialization_plan(
    const PipelineStageDesc& desc,
    const RuntimeStageExecutableDescriptor& stage_descriptor) {
    PipelineStageMaterializationPlan plan;
    plan.kind = PipelineStageMaterializationKind::SingleStage;
    static_cast<PipelineStageIoPlan&>(plan.io_plan) =
        static_cast<const PipelineStageIoPlan&>(desc);
    plan.io_plan.runtime_stage_index = desc.runtime_stage_index;
    if (plan.io_plan.stage_name.empty()) {
        plan.io_plan.stage_name = stage_descriptor.stage_name;
    }
    if (plan.io_plan.op_family.empty()) {
        plan.io_plan.op_family = stage_descriptor.op_family;
    }
    plan.descriptor_stage_index = desc.runtime_stage_index;
    plan.materialized_descriptor = stage_descriptor;
    plan.materialized_descriptor_valid = true;
    return plan;
}

} // namespace detail

inline std::shared_ptr<const RuntimeExecutionPlan>
make_runtime_execution_plan(
    std::shared_ptr<RuntimeExecutableDescriptor> runtime_descriptor,
    std::vector<PipelineStageDesc> descs) {
    OPENVINO_ASSERT(runtime_descriptor,
                    "GFX unit test: runtime descriptor is null");
    runtime_descriptor->materialization_finalized = false;
    runtime_descriptor->materialization_stages.clear();
    runtime_descriptor->materialization_stages.reserve(descs.size());
    for (const auto& desc : descs) {
        OPENVINO_ASSERT(
            desc.runtime_stage_index != PipelineStageDesc::npos &&
                desc.runtime_stage_index < runtime_descriptor->stages.size(),
            "GFX unit test: runtime plan descriptor index is invalid");
        auto& stage_descriptor =
            runtime_descriptor->stages[desc.runtime_stage_index];
        detail::complete_stage_descriptor(stage_descriptor, desc);
        runtime_descriptor->materialization_stages.push_back(
            detail::make_materialization_plan(desc, stage_descriptor));
    }
    runtime_descriptor->materialization_finalized = true;

    detail::RuntimePlanStageFactory stage_factory(std::move(descs));
    RuntimeExecutionPlanBuildRequest request;
    request.stage_factory = &stage_factory;
    request.runtime_descriptor = std::move(runtime_descriptor);
    return RuntimeExecutionPlan::build(std::move(request));
}

} // namespace test
} // namespace gfx_plugin
} // namespace ov
