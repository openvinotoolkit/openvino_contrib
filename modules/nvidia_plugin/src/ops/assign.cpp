// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "assign.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/op/util/variable_extension.hpp>

namespace ov {
namespace nvidia_gpu {

AssignOp::AssignOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto& var_ext = dynamic_cast<const ov::op::util::VariableExtension&>(node);
    variable_id_ = var_ext.get_variable_id();
    element_type_ = node.get_input_element_type(0);

    const auto& pshape = node.get_input_partial_shape(0);
    is_dynamic_ = pshape.is_dynamic();
    if (!is_dynamic_) {
        static_input_shape_ = pshape.to_shape();
    }
}

void AssignOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Assign expects 1 input, got: ", inputs.size());
    OPENVINO_ASSERT(context.hasVariableContext(), "Assign requires VariableContext");

    auto& varCtx = context.getVariableContext();
    auto state = varCtx.get_variable_state(variable_id_);
    const auto& stream = context.getThreadContext().stream();

    ov::Shape shape;
    if (is_dynamic_) {
        auto& shapeCtx = const_cast<InferenceRequestContext&>(context).getShapeContext();
        auto input_buf_id = input_ids_[0].GetBuffer().GetId();
        if (shapeCtx.hasShape(input_buf_id)) {
            shape = shapeCtx.getShape(input_buf_id);
        } else {
            shape = static_input_shape_;
        }
    } else {
        shape = static_input_shape_;
    }

    state->update_device_state(stream,
                               CUDA::DevicePointer<const void*>{inputs[0].get()},
                               shape);
}

OPERATION_REGISTER(AssignOp, Assign);

}  // namespace nvidia_gpu
}  // namespace ov
