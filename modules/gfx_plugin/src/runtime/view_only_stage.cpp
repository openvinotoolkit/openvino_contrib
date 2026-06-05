// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/view_only_stage.hpp"

#include <cctype>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_compiler_owned_view_descriptor(
    const RuntimeStageExecutableDescriptor& descriptor) {
    return descriptor.origin == KernelArtifactOrigin::Metadata &&
           descriptor.payload_kind == KernelArtifactPayloadKind::None &&
           descriptor.kernel_id == "metadata" &&
           descriptor.tensor_view_only;
}

std::string descriptor_stage_name(
    const RuntimeStageExecutableDescriptor& descriptor) {
    if (!descriptor.output_bindings.empty() &&
        !descriptor.output_bindings.front().logical_name.empty()) {
        return descriptor.output_bindings.front().logical_name;
    }
    if (!descriptor.stage_name.empty()) {
        return descriptor.stage_name;
    }
    if (!descriptor.manifest_ref.empty()) {
        return descriptor.manifest_ref;
    }
    if (!descriptor.kernel_id.empty()) {
        return descriptor.kernel_id;
    }
    return "view_only";
}

std::string descriptor_stage_type(
    const RuntimeStageExecutableDescriptor& descriptor) {
    return descriptor.op_family.empty() ? std::string{"ViewOnly"}
                                        : descriptor.op_family;
}

ov::element::Type element_type_from_contract(const std::string& name) {
    if (name == "f32" || name == "float32") {
        return ov::element::f32;
    }
    if (name == "f16" || name == "float16") {
        return ov::element::f16;
    }
    if (name == "bf16") {
        return ov::element::bf16;
    }
    if (name == "i64") {
        return ov::element::i64;
    }
    if (name == "i32") {
        return ov::element::i32;
    }
    if (name == "i16") {
        return ov::element::i16;
    }
    if (name == "i8") {
        return ov::element::i8;
    }
    if (name == "u64") {
        return ov::element::u64;
    }
    if (name == "u32") {
        return ov::element::u32;
    }
    if (name == "u16") {
        return ov::element::u16;
    }
    if (name == "u8") {
        return ov::element::u8;
    }
    if (name == "boolean" || name == "bool") {
        return ov::element::boolean;
    }
    return ov::element::dynamic;
}

bool consume_whitespace(const std::string& text, size_t& pos) {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos < text.size();
}

bool parse_static_shape_contract(const std::string& text, ov::Shape& shape) {
    shape.clear();
    size_t pos = 0;
    if (!consume_whitespace(text, pos) || text[pos] != '{') {
        return false;
    }
    ++pos;
    consume_whitespace(text, pos);
    if (pos < text.size() && text[pos] == '}') {
        ++pos;
        consume_whitespace(text, pos);
        return pos == text.size();
    }
    while (pos < text.size()) {
        consume_whitespace(text, pos);
        if (pos >= text.size() ||
            !std::isdigit(static_cast<unsigned char>(text[pos]))) {
            return false;
        }
        size_t next = pos;
        while (next < text.size() &&
               std::isdigit(static_cast<unsigned char>(text[next]))) {
            ++next;
        }
        shape.push_back(static_cast<size_t>(std::stoull(text.substr(pos, next - pos))));
        pos = next;
        consume_whitespace(text, pos);
        if (pos >= text.size()) {
            return false;
        }
        if (text[pos] == '}') {
            ++pos;
            consume_whitespace(text, pos);
            return pos == text.size();
        }
        if (text[pos] != ',') {
            return false;
        }
        ++pos;
    }
    return false;
}

class ViewOnlyStage final : public GpuStage {
public:
    explicit ViewOnlyStage(RuntimeStageExecutableDescriptor descriptor)
        : m_descriptor(std::move(descriptor)),
          m_name(descriptor_stage_name(m_descriptor)),
          m_type(descriptor_stage_type(m_descriptor)) {
        OPENVINO_ASSERT(is_compiler_owned_view_descriptor(m_descriptor),
                        "GFX view-only stage requires a compiler-owned metadata descriptor");
    }

    void init(GpuBufferManager*) override {}

    void prepare_runtime_handle(GpuBufferManager*) override {}

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

    std::unique_ptr<GpuStage> clone() const override {
        return std::make_unique<ViewOnlyStage>(m_descriptor);
    }

private:
    ov::Shape output_shape(size_t output_idx) const {
        if (output_idx < m_descriptor.output_bindings.size()) {
            ov::Shape shape;
            if (parse_static_shape_contract(
                    m_descriptor.output_bindings[output_idx].partial_shape,
                    shape)) {
                return shape;
            }
        }
        if (output_idx == 0 && !m_inputs.empty() && m_inputs.front()) {
            return m_inputs.front()->shape;
        }
        return {};
    }

    ov::element::Type output_type(size_t output_idx) const {
        if (output_idx < m_descriptor.output_bindings.size()) {
            return element_type_from_contract(
                m_descriptor.output_bindings[output_idx].element_type);
        }
        return ov::element::dynamic;
    }

    RuntimeStageExecutableDescriptor m_descriptor;
    std::string m_name;
    std::string m_type;
    std::vector<GpuTensor*> m_inputs;
    std::vector<GpuTensor*> m_outputs;
};

}  // namespace

std::unique_ptr<GpuStage> create_view_only_stage(
    const RuntimeStageExecutableDescriptor& descriptor) {
    if (!is_compiler_owned_view_descriptor(descriptor)) {
        return {};
    }
    return std::make_unique<ViewOnlyStage>(descriptor);
}

}  // namespace gfx_plugin
}  // namespace ov
