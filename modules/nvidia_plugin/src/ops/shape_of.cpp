// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

ShapeOfOp::ShapeOfOp(const CreationContext& context,
                     const ov::Node& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    output_type_ = node.get_output_element_type(0);
    OPENVINO_ASSERT(output_type_ == ov::element::i32 || output_type_ == ov::element::i64,
                    "ShapeOf output type must be i32 or i64, got: ", output_type_);

    const auto& pshape = node.get_input_partial_shape(0);
    is_dynamic_ = pshape.is_dynamic();
    if (pshape.rank().is_static()) {
        input_rank_ = pshape.rank().get_length();
    } else {
        input_rank_ = 0;
    }
    if (!is_dynamic_) {
        static_input_shape_ = pshape.to_shape();
    }
}

void ShapeOfOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    OPENVINO_ASSERT(outputs.size() == 1, "ShapeOf expects 1 output");
    const auto& stream = context.getThreadContext().stream();

    ov::Shape input_shape;
    if (is_dynamic_) {
        auto& dynBufCtx = const_cast<InferenceRequestContext&>(context).getDynamicBufferContext();
        auto input_buf_id = input_ids_[0].GetBuffer().GetId();
        // static_input_shape_ is only populated for static inputs, so it must not be
        // used as a fallback here: the runtime shape must have been registered by the
        // producing op. Fail loudly instead of emitting a wrong rank-0 result.
        OPENVINO_ASSERT(dynBufCtx.hasShape(input_buf_id),
                        "ShapeOf: runtime input shape is not available in the dynamic buffer context");
        input_shape = dynBufCtx.getShape(input_buf_id);
    } else {
        input_shape = static_input_shape_;
    }

    auto rank = input_shape.size();

    if (output_type_ == ov::element::i64) {
        std::vector<int64_t> shape_data(rank);
        for (size_t i = 0; i < rank; ++i) {
            shape_data[i] = static_cast<int64_t>(input_shape[i]);
        }
        stream.upload(outputs[0], shape_data.data(), rank * sizeof(int64_t));
    } else {
        std::vector<int32_t> shape_data(rank);
        for (size_t i = 0; i < rank; ++i) {
            shape_data[i] = static_cast<int32_t>(input_shape[i]);
        }
        stream.upload(outputs[0], shape_data.data(), rank * sizeof(int32_t));
    }
}

OPERATION_REGISTER(ShapeOfOp, ShapeOf);

}  // namespace nvidia_gpu
}  // namespace ov
