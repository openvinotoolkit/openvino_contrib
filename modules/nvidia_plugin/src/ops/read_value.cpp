// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "read_value.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/op/util/variable_extension.hpp>

namespace ov {
namespace nvidia_gpu {

ReadValueOp::ReadValueOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto& var_ext = dynamic_cast<const ov::op::util::VariableExtension&>(node);
    variable_id_ = var_ext.get_variable_id();
    element_type_ = node.get_output_element_type(0);
    has_init_value_ = node.get_input_size() > 0;

    const auto& pshape = node.get_output_partial_shape(0);
    if (pshape.is_static()) {
        output_byte_size_ = element_type_.size() * ov::shape_size(pshape.to_shape());
    } else {
        output_byte_size_ = 0;
    }
}

void ReadValueOp::Execute(const InferenceRequestContext& context,
                          Inputs inputs,
                          Outputs outputs,
                          const Workbuffers&) const {
    OPENVINO_ASSERT(outputs.size() == 1, "ReadValue expects 1 output, got: ", outputs.size());
    OPENVINO_ASSERT(context.hasVariableContext(), "ReadValue requires VariableContext");

    auto& varCtx = context.getVariableContext();
    auto state = varCtx.get_variable_state(variable_id_);
    const auto& stream = context.getThreadContext().stream();

    if (state->is_reset_state()) {
        if (has_init_value_ && !inputs.empty()) {
            // First inference or after reset: use init_value input
            auto byte_size = output_byte_size_;
            if (byte_size == 0) {
                // Dynamic shape: get size from state or init value
                byte_size = state->device_buffer_byte_size();
            }
            if (byte_size > 0) {
                stream.transfer(outputs[0],
                                CUDA::DevicePointer<const void*>{inputs[0].get()},
                                byte_size);
            }
        } else {
            // No init_value: zero-fill output
            auto byte_size = output_byte_size_;
            if (byte_size == 0) {
                byte_size = state->device_buffer_byte_size();
            }
            if (byte_size > 0) {
                stream.memset(outputs[0], 0, byte_size);
            }
        }
    } else {
        // Normal case: copy state buffer to output (D2D)
        auto byte_size = state->device_buffer_byte_size();
        if (byte_size > 0) {
            state->read_device_state(stream, outputs[0], byte_size);
        }
    }
}

OPERATION_REGISTER(ReadValueOp, ReadValue);

}  // namespace nvidia_gpu
}  // namespace ov
