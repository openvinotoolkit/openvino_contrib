// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/scatter_elements_update.hpp>

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

ScatterElementsUpdateOp::ScatterElementsUpdateOp(const CreationContext& context,
                                                 const ov::Node& node,
                                                 IndexCollection&& inputIds,
                                                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 4, "Node name: ", GetName());  // data, indices, updates, axis
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    // opset12 adds a reduction; only the default (none) is supported here, which
    // is also what the v12 -> v3 downgrade produces.
    if (const auto v12 = dynamic_cast<const ov::op::v12::ScatterElementsUpdate*>(&node)) {
        OPENVINO_ASSERT(v12->get_reduction() == ov::op::v12::ScatterElementsUpdate::Reduction::NONE,
                        "Node name: ",
                        GetName(),
                        "; ScatterElementsUpdate reduction modes are not supported");
    }

    element_size_ = node.get_input_element_type(0).size();
    OPENVINO_ASSERT(element_size_ == 1 || element_size_ == 2 || element_size_ == 4 || element_size_ == 8,
                    "Node name: ",
                    GetName(),
                    "; ScatterElementsUpdate unsupported element size ",
                    element_size_);
    OPENVINO_ASSERT(node.get_input_element_type(2) == node.get_input_element_type(0), "Node name: ", GetName());

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i32 && indices_type != ov::element::Type_t::i64) {
        throw_ov_exception(
            fmt::format("Index element type = {} is not supported by ScatterElementsUpdate operation!", indices_type));
    }
    index_size_ = node.get_input_element_type(1).size();

    const auto& data_shape = node.get_input_shape(0);
    const auto& updates_shape = node.get_input_shape(2);
    const size_t rank = data_shape.size();
    OPENVINO_ASSERT(rank == updates_shape.size() && rank > 0, "Node name: ", GetName());
    OPENVINO_ASSERT(rank <= kernel::ScatterElementsUpdateParams::kMaxRank, "Node name: ", GetName());

    const auto axis_constant = ov::as_type_ptr<const ov::op::v0::Constant>(node.get_input_node_shared_ptr(3));
    OPENVINO_ASSERT(axis_constant, "Node name: ", GetName(), "; ScatterElementsUpdate axis must be a Constant");
    int64_t axis = axis_constant->cast_vector<int64_t>().at(0);
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(rank), "Node name: ", GetName());

    const auto data_strides = rowMajorStrides(data_shape);
    const auto updates_strides = rowMajorStrides(updates_shape);

    params_.rank = static_cast<int>(rank);
    params_.axis = static_cast<int>(axis);
    params_.updates_axis_size = updates_shape[axis];
    params_.data_axis_dim = static_cast<long long>(data_shape[axis]);
    params_.num_data_elements = ov::shape_size(data_shape);
    const size_t updates_total = ov::shape_size(updates_shape);
    params_.num_rows = params_.updates_axis_size == 0 ? 0 : updates_total / params_.updates_axis_size;
    for (size_t i = 0; i < rank; ++i) {
        params_.updates_shape[i] = updates_shape[i];
        params_.updates_strides[i] = updates_strides[i];
        params_.data_strides[i] = data_strides[i];
    }
}

void ScatterElementsUpdateOp::Execute(const InferenceRequestContext& context,
                                      Inputs inputs,
                                      Outputs outputs,
                                      const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    kernel::scatter_elements_update(context.getThreadContext().stream().get(),
                                    params_,
                                    element_size_,
                                    index_size_,
                                    inputs[0].get(),
                                    inputs[1].get(),
                                    inputs[2].get(),
                                    outputs[0].get());
}

CudaGraphCompatibility ScatterElementsUpdateOp::GetCudaGraphCompatibilityImpl() const {
    return CudaGraphCompatibility::FULL;
}

OPERATION_REGISTER(ScatterElementsUpdateOp, ScatterElementsUpdate);

}  // namespace nvidia_gpu
}  // namespace ov
