// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "infer_io_utils.hpp"

#include <cstdint>
#include <limits>
#include <vector>

#include "openvino/core/except.hpp"
#include "runtime/gfx_runtime_value_limits.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

template <typename T>
std::vector<int64_t> copy_integral_tensor_values(const ov::Tensor& tensor,
                                                 size_t count) {
    std::vector<int64_t> values;
    values.reserve(count);
    const auto* data = tensor.data<const T>();
    for (size_t i = 0; i < count; ++i) {
        values.push_back(static_cast<int64_t>(data[i]));
    }
    return values;
}

std::vector<int64_t> inline_runtime_i64_values(const ov::Tensor& tensor) {
    const size_t count = tensor.get_size();
    if (count == 0 || count > kGfxInlineRuntimeI64ValueLimit) {
        return {};
    }

    const auto type = tensor.get_element_type();
    if (type == ov::element::i64) {
        return copy_integral_tensor_values<int64_t>(tensor, count);
    }
    if (type == ov::element::i32) {
        return copy_integral_tensor_values<int32_t>(tensor, count);
    }
    if (type == ov::element::i16) {
        return copy_integral_tensor_values<int16_t>(tensor, count);
    }
    if (type == ov::element::i8) {
        return copy_integral_tensor_values<int8_t>(tensor, count);
    }
    if (type == ov::element::u32) {
        return copy_integral_tensor_values<uint32_t>(tensor, count);
    }
    if (type == ov::element::u16) {
        return copy_integral_tensor_values<uint16_t>(tensor, count);
    }
    if (type == ov::element::u8) {
        return copy_integral_tensor_values<uint8_t>(tensor, count);
    }
    if (type == ov::element::u64) {
        const auto* data = tensor.data<const uint64_t>();
        std::vector<int64_t> values;
        values.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            OPENVINO_ASSERT(data[i] <=
                                static_cast<uint64_t>(
                                    std::numeric_limits<int64_t>::max()),
                            "GFX: runtime metadata tensor value exceeds i64 "
                            "range");
            values.push_back(static_cast<int64_t>(data[i]));
        }
        return values;
    }
    return {};
}

}  // namespace

HostInputBinding prepare_host_input_binding(const ov::Tensor& host,
                                            GpuBackend backend,
                                            const char* error_prefix) {
    OPENVINO_ASSERT(host && host.data(), error_prefix, ": input host tensor is empty");
    HostInputBinding binding{};
    binding.bytes = host.get_byte_size();
    binding.tensor.shape = host.get_shape();
    binding.tensor.expected_type = host.get_element_type();
    binding.tensor.buf.type = host.get_element_type();
    binding.tensor.buf.backend = backend;
    binding.tensor.i64_values = inline_runtime_i64_values(host);
    return binding;
}

void prepare_reusable_host_output_plan(PreparedInferHostOutputPlan& plan,
                                       const PreparedInferOutputPlan& output_plan,
                                       const std::vector<ov::Tensor>& bound_output_hosts) {
    if (plan.outputs.size() != output_plan.outputs.size()) {
        plan.outputs.assign(output_plan.outputs.size(), {});
    }

    for (size_t idx = 0; idx < output_plan.outputs.size(); ++idx) {
        auto& prepared_host = plan.outputs[idx];
        const auto& prepared_output = output_plan.outputs[idx];
        const bool has_bound_host = idx < bound_output_hosts.size() && bound_output_hosts[idx];
        if (has_bound_host || prepared_output.static_shape.empty() ||
            prepared_output.static_type == ov::element::dynamic) {
            prepared_host.shape.clear();
            prepared_host.type = ov::element::dynamic;
            prepared_host.host = {};
            continue;
        }

        prepared_host.shape = prepared_output.static_shape;
        prepared_host.type = prepared_output.static_type;
        if (!prepared_host.host ||
            prepared_host.host.get_shape() != prepared_host.shape ||
            prepared_host.host.get_element_type() != prepared_host.type) {
            prepared_host.host = ov::Tensor(prepared_host.type, prepared_host.shape);
        }
    }
}

HostOutputBinding prepare_host_output_binding(const OutputViewInfo& info,
                                              const ov::Tensor* host_override,
                                              ov::Tensor* reusable_host) {
    HostOutputBinding binding{};
    binding.bytes = tensor_byte_size(info.shape, info.type);
    if (host_override && *host_override) {
        binding.host = *host_override;
    } else if (reusable_host) {
        if (!*reusable_host ||
            reusable_host->get_element_type() != info.type ||
            reusable_host->get_shape() != info.shape) {
            *reusable_host = ov::Tensor(info.type, info.shape);
        }
        binding.host = *reusable_host;
    } else {
        binding.host = ov::Tensor(info.type, info.shape);
    }
    return binding;
}

bool init_stage_output_desc(GpuBackend backend,
                            InferStage& stage,
                            size_t out_idx,
                            GpuTensor& out_ref,
                            GpuBufferDesc& desc,
                            bool is_model_output,
                            bool skip_view_ops,
                            const char* error_prefix) {
    (void)backend;
    if (skip_view_ops && is_view_op(stage)) {
        return false;
    }
    const auto out_shape = ensure_stage_output_shape(stage, out_idx);
    if (out_shape.empty()) {
        return false;
    }
    const auto et = resolve_stage_output_type(stage, out_ref, out_idx, error_prefix);
    size_t bytes = tensor_byte_size(out_shape, et);

    desc.bytes = bytes;
    desc.type = et;
    if (const auto* descriptor = runtime_stage_descriptor_or_null(stage);
        descriptor && !descriptor->stage_name.empty()) {
        desc.label = descriptor->stage_name.c_str();
    } else if (stage.stage) {
        desc.label = stage.stage->name().c_str();
    } else {
        desc.label = nullptr;
    }
    if (apply_runtime_output_memory_contract(stage,
                                             out_idx,
                                             desc,
                                             out_ref,
                                             error_prefix)) {
        return true;
    }

    const bool prefer_private = !is_model_output;
    out_ref.prefer_private = prefer_private;
    desc.cpu_read = !prefer_private;
    desc.cpu_write = !prefer_private;
    desc.prefer_device_local = prefer_private;
    desc.usage = is_model_output ? BufferUsage::IO : BufferUsage::Intermediate;
    return true;
}

}  // namespace gfx_plugin
}  // namespace ov
