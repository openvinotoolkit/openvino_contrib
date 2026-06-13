// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/tile.hpp>

namespace ov {
namespace nvidia_gpu {

namespace {
// Row-major strides of `shape` (stride of the last axis is 1).
std::vector<size_t> rowMajorStrides(const ov::Shape& shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] = strides[i] * shape[i];
    }
    return strides;
}
}  // namespace

TileOp::TileOp(const CreationContext& context,
               const ov::Node& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());  // data, repeats
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const auto& in_shape = node.get_input_shape(0);
    const auto& out_shape = node.get_output_shape(0);
    const size_t rank = out_shape.size();
    OPENVINO_ASSERT(rank <= kernel::TileParams::kMaxRank,
                    "Node name: ",
                    GetName(),
                    "; Tile supports up to ",
                    kernel::TileParams::kMaxRank,
                    " dims");
    OPENVINO_ASSERT(in_shape.size() <= rank, "Node name: ", GetName());

    element_size_ = node.get_input_element_type(0).size();
    OPENVINO_ASSERT(element_size_ == 1 || element_size_ == 2 || element_size_ == 4 || element_size_ == 8,
                    "Node name: ",
                    GetName(),
                    "; Tile unsupported element size ",
                    element_size_);

    // Left-pad the input shape with 1s up to the output rank (opset Tile aligns
    // ranks by prepending unit dimensions to the data).
    ov::Shape padded_in(rank, 1);
    const size_t offset = rank - in_shape.size();
    for (size_t i = 0; i < in_shape.size(); ++i) {
        padded_in[offset + i] = in_shape[i];
    }
    const auto in_strides = rowMajorStrides(padded_in);
    const auto out_strides = rowMajorStrides(out_shape);

    params_.rank = static_cast<int>(rank);
    params_.num_output_elements =
        std::accumulate(out_shape.begin(), out_shape.end(), size_t{1}, std::multiplies<size_t>());
    for (size_t i = 0; i < rank; ++i) {
        params_.input_shape[i] = padded_in[i];
        params_.input_strides[i] = in_strides[i];
        params_.output_strides[i] = out_strides[i];
    }
}

void TileOp::Execute(const InferenceRequestContext& context,
                     Inputs inputs,
                     Outputs outputs,
                     const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    kernel::tile(context.getThreadContext().stream().get(), params_, element_size_, inputs[0].get(), outputs[0].get());
}

CudaGraphCompatibility TileOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(TileOp, Tile);

}  // namespace nvidia_gpu
}  // namespace ov
