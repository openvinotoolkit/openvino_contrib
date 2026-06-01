// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/view_only_stage.hpp"

#include <string>
#include <utility>
#include <vector>

#include "compiler/executable_bundle.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_compiler_owned_view_descriptor(
    const RuntimeStageExecutableDescriptor* descriptor) {
    return descriptor &&
           descriptor->origin == compiler::KernelArtifactOrigin::Metadata &&
           descriptor->payload_kind == compiler::KernelArtifactPayloadKind::None &&
           descriptor->kernel_id == "metadata" &&
           descriptor->tensor_view_only;
}

class ViewOnlyStage final : public GpuStage {
public:
    ViewOnlyStage(std::shared_ptr<const ov::Node> node,
                  RuntimeStageExecutableDescriptor descriptor)
        : m_node(std::move(node)),
          m_descriptor(std::move(descriptor)),
          m_name(m_node ? m_node->get_friendly_name() : std::string{}),
          m_type(m_node ? m_node->get_type_name() : std::string{}) {
        OPENVINO_ASSERT(m_node, "GFX view-only stage requires a node");
        OPENVINO_ASSERT(is_compiler_owned_view_descriptor(&m_descriptor),
                        "GFX view-only stage requires a compiler-owned metadata descriptor");
    }

    void init(GpuBufferManager*) override {}

    void compile(GpuBufferManager*) override {}

    void execute(GpuCommandBufferHandle) override {
        OPENVINO_ASSERT(!m_inputs.empty() && m_inputs.front() &&
                            m_inputs.front()->buf.valid(),
                        "GFX view-only stage input is not materialized for ",
                        m_name);
        OPENVINO_ASSERT(!m_outputs.empty(),
                        "GFX view-only stage output is not bound for ",
                        m_name);

        auto* input = m_inputs.front();
        for (size_t output_idx = 0; output_idx < m_outputs.size(); ++output_idx) {
            auto* output = m_outputs[output_idx];
            OPENVINO_ASSERT(output,
                            "GFX view-only stage output ",
                            output_idx,
                            " is null for ",
                            m_name);
            output->buf = input->buf;
            output->buf.external = true;
            output->buf.owned = false;
            output->shape = output_shape(output_idx);
            output->expected_type = output_type(output_idx);
            output->gqa_broadcast_view = input->gqa_broadcast_view;
            output->gqa_storage_shape = input->gqa_storage_shape;
            output->gqa_kv_heads = input->gqa_kv_heads;
            if (!input->i64_values.empty() &&
                input->i64_values.size() == ov::shape_size(output->shape)) {
                output->i64_values = input->i64_values;
            } else {
                output->i64_values.clear();
            }
        }
    }

    void set_inputs(const std::vector<GpuTensor*>& inputs) override {
        m_inputs = inputs;
    }

    void set_output(GpuTensor* output) override {
        m_outputs.clear();
        if (output) {
            m_outputs.push_back(output);
        }
    }

    void set_output_refs(const std::vector<GpuTensor*>& outputs) override {
        m_outputs = outputs;
    }

    const std::string& name() const override {
        return m_name;
    }

    const std::string& type() const override {
        return m_type;
    }

    bool is_view_only() const override {
        return true;
    }

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<ViewOnlyStage>(m_node, m_descriptor);
    }

private:
    ov::Shape output_shape(size_t output_idx) const {
        if (output_idx < m_node->get_output_size() &&
            m_node->get_output_partial_shape(output_idx).is_static()) {
            return m_node->get_output_shape(output_idx);
        }
        if (output_idx == 0 && !m_inputs.empty() && m_inputs.front()) {
            return m_inputs.front()->shape;
        }
        return {};
    }

    ov::element::Type output_type(size_t output_idx) const {
        if (output_idx < m_node->get_output_size()) {
            return m_node->get_output_element_type(output_idx);
        }
        return ov::element::dynamic;
    }

    std::shared_ptr<const ov::Node> m_node;
    RuntimeStageExecutableDescriptor m_descriptor;
    std::string m_name;
    std::string m_type;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
};

}  // namespace

std::unique_ptr<GpuStage> create_view_only_stage(
    const std::shared_ptr<const ov::Node>& node,
    const RuntimeStageExecutableDescriptor* descriptor) {
    if (!node || !is_compiler_owned_view_descriptor(descriptor)) {
        return {};
    }
    return std::make_unique<ViewOnlyStage>(node, *descriptor);
}

}  // namespace gfx_plugin
}  // namespace ov
