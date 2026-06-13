// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/gather_nd.hpp>

namespace ov {
namespace nvidia_gpu {

namespace {
size_t product(const ov::Shape& shape, size_t from, size_t to) {  // [from, to)
    return std::accumulate(shape.begin() + from, shape.begin() + to, size_t{1}, std::multiplies<size_t>());
}
}  // namespace

GatherNDOp::GatherNDOp(const CreationContext& context,
                       const ov::Node& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto gather = dynamic_cast<const ov::op::util::GatherNDBase*>(&node);
    OPENVINO_ASSERT(gather, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());  // data, indices
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    element_size_ = node.get_input_element_type(0).size();
    OPENVINO_ASSERT(element_size_ == 1 || element_size_ == 2 || element_size_ == 4 || element_size_ == 8,
                    "Node name: ",
                    GetName(),
                    "; GatherND unsupported element size ",
                    element_size_);

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i32 && indices_type != ov::element::Type_t::i64) {
        throw_ov_exception(fmt::format("Index element type = {} is not supported by GatherND operation!", indices_type));
    }
    index_size_ = node.get_input_element_type(1).size();

    const auto& data_shape = node.get_input_shape(0);
    const auto& indices_shape = node.get_input_shape(1);
    const size_t data_rank = data_shape.size();
    const size_t indices_rank = indices_shape.size();
    const size_t batch_dims = gather->get_batch_dims();
    const size_t index_len = indices_shape.back();
    OPENVINO_ASSERT(index_len <= kernel::GatherNDParams::kMaxIndexLen,
                    "Node name: ",
                    GetName(),
                    "; GatherND index tuple length exceeds ",
                    kernel::GatherNDParams::kMaxIndexLen);
    OPENVINO_ASSERT(batch_dims + index_len <= data_rank, "Node name: ", GetName());

    params_.batch_size = product(data_shape, 0, batch_dims);
    // index tuples per batch = product of the indices' middle dims (between the
    // batch dims and the trailing index-length dim).
    params_.num_index_tuples = product(indices_shape, batch_dims, indices_rank - 1);
    params_.slice_size = product(data_shape, batch_dims + index_len, data_rank);
    params_.batch_block_size = product(data_shape, batch_dims, data_rank);
    params_.index_len = static_cast<int>(index_len);
    params_.num_output_elements = ov::shape_size(node.get_output_shape(0));
    for (size_t l = 0; l < index_len; ++l) {
        params_.indexed_dim[l] = static_cast<long long>(data_shape[batch_dims + l]);
        params_.indexed_stride[l] = product(data_shape, batch_dims + l + 1, data_rank);
    }
}

void GatherNDOp::Execute(const InferenceRequestContext& context,
                         Inputs inputs,
                         Outputs outputs,
                         const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    kernel::gather_nd(context.getThreadContext().stream().get(),
                      params_,
                      element_size_,
                      index_size_,
                      inputs[0].get(),
                      inputs[1].get(),
                      outputs[0].get());
}

CudaGraphCompatibility GatherNDOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(GatherNDOp, GatherND);

}  // namespace nvidia_gpu
}  // namespace ov
