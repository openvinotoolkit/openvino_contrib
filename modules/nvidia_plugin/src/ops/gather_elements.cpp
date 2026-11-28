// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/op/gather_elements.hpp>

namespace ov {
namespace nvidia_gpu {

namespace {
std::vector<size_t> rowMajorStrides(const ov::Shape& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}
}  // namespace

GatherElementsOp::GatherElementsOp(const CreationContext& context,
                                   const ov::Node& node,
                                   IndexCollection&& inputIds,
                                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto gather = dynamic_cast<const ov::op::v6::GatherElements*>(&node);
    OPENVINO_ASSERT(gather, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());  // data, indices
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    element_size_ = node.get_input_element_type(0).size();
    OPENVINO_ASSERT(element_size_ == 1 || element_size_ == 2 || element_size_ == 4 || element_size_ == 8,
                    "Node name: ",
                    GetName(),
                    "; GatherElements unsupported element size ",
                    element_size_);

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i32 && indices_type != ov::element::Type_t::i64) {
        throw_ov_exception(
            fmt::format("Index element type = {} is not supported by GatherElements operation!", indices_type));
    }
    index_size_ = node.get_input_element_type(1).size();

    const auto& data_shape = node.get_input_shape(0);
    const auto& indices_shape = node.get_input_shape(1);  // == output shape
    const size_t rank = data_shape.size();
    OPENVINO_ASSERT(rank == indices_shape.size() && rank > 0, "Node name: ", GetName());
    OPENVINO_ASSERT(rank <= kernel::GatherElementsParams::kMaxRank, "Node name: ", GetName());

    int64_t axis = gather->get_axis();
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(rank), "Node name: ", GetName());

    const auto data_strides = rowMajorStrides(data_shape);
    const auto indices_strides = rowMajorStrides(indices_shape);

    params_.rank = static_cast<int>(rank);
    params_.axis = static_cast<int>(axis);
    params_.data_axis_dim = static_cast<long long>(data_shape[axis]);
    params_.num_output_elements = ov::shape_size(indices_shape);
    for (size_t i = 0; i < rank; ++i) {
        params_.indices_strides[i] = indices_strides[i];
        params_.data_strides[i] = data_strides[i];
    }
}

void GatherElementsOp::Execute(const InferenceRequestContext& context,
                               Inputs inputs,
                               Outputs outputs,
                               const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    kernel::gather_elements(context.getThreadContext().stream().get(),
                            params_,
                            element_size_,
                            index_size_,
                            inputs[0].get(),
                            inputs[1].get(),
                            outputs[0].get());
}

CudaGraphCompatibility GatherElementsOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(GatherElementsOp, GatherElements);

}  // namespace nvidia_gpu
}  // namespace ov
