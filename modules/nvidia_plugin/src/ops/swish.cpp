// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swish.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <utility>
#include <vector>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {
double beta_from_constant(const ov::Node& swish_node) {
    constexpr auto tensor_index = 1;
    constexpr auto default_value = 1.0;
    if (tensor_index >= swish_node.get_input_size()) {
        return default_value;
    }
    const ov::Node* constant_node = swish_node.get_input_node_ptr(tensor_index);
    const ov::op::v0::Constant* constant = dynamic_cast<const ov::op::v0::Constant*>(constant_node);
    OPENVINO_ASSERT(constant);
    switch (constant->get_output_element_type(0)) {
        case ov::element::Type_t::f16:
            return *constant->get_data_ptr<ov::float16>();
        case ov::element::Type_t::f32:
            return *constant->get_data_ptr<float>();
        case ov::element::Type_t::f64:
            return *constant->get_data_ptr<double>();
        default:
            OPENVINO_ASSERT(false);
    }
}
}  // namespace

SwishOp::SwishOp(const CreationContext& context,
                 const ov::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 1 || node.get_input_size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());
    const auto input_element_type = node.get_input_element_type(0);
    const auto output_element_type = node.get_output_element_type(0);
    OPENVINO_ASSERT(input_element_type == output_element_type, "Node name: ", GetName());
    const auto input_shape = node.get_input_shape(0);
    const auto output_shape = node.get_output_shape(0);
    OPENVINO_ASSERT(input_shape == output_shape, "Node name: ", GetName());
    size_t num_elements = ov::shape_size(input_shape);
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    const double beta = beta_from_constant(node);
    kernel_ = kernel::Swish{
        convertDataType<ov::nvidia_gpu::kernel::Type_t>(input_element_type), max_threads_per_block, num_elements, beta};
}

void SwishOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputTensors.size() >= 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputTensors.size() == 1, "Node name: ", GetName());
    const auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(), inputTensors[0].get(), outputTensors[0].get());
}

CudaGraphCompatibility SwishOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(SwishOp, Swish);
}  // namespace nvidia_gpu
}  // namespace ov
