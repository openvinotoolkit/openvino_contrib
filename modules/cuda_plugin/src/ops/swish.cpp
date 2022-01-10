// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swish.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <utility>
#include <vector>

#include "converters.hpp"
#include "ngraph/op/constant.hpp"

namespace CUDAPlugin {

namespace {
double beta_from_constant(const ngraph::Node& swish_node) {
    constexpr auto tensor_index = 1;
    constexpr auto default_value = 1.0;
    if (tensor_index >= swish_node.get_input_size()) {
        return default_value;
    }
    const ngraph::Node* constant_node = swish_node.get_input_node_ptr(tensor_index);
    const ngraph::op::v0::Constant* constant = dynamic_cast<const ngraph::op::v0::Constant*>(constant_node);
    Expects(constant);
    switch (constant->get_output_element_type(0)) {
        case ngraph::element::Type_t::f16:
            return *constant->get_data_ptr<ngraph::float16>();
        case ngraph::element::Type_t::f32:
            return *constant->get_data_ptr<float>();
        case ngraph::element::Type_t::f64:
            return *constant->get_data_ptr<double>();
        default:
            Expects(false);
    }
}
}  // namespace

SwishOp::SwishOp(const CreationContext& context,
                 const ngraph::Node& node,
                 IndexCollection&& inputIds,
                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Expects(node.get_input_size() == 1 || node.get_input_size() == 2);
    Expects(node.get_output_size() == 1);
    const auto input_element_type = node.get_input_element_type(0);
    const auto output_element_type = node.get_output_element_type(0);
    Expects(input_element_type == output_element_type);
    const auto input_shape = node.get_input_shape(0);
    const auto output_shape = node.get_output_shape(0);
    Expects(input_shape == output_shape);
    size_t num_elements = ngraph::shape_size(input_shape);
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    const double beta = beta_from_constant(node);
    kernel_ = kernel::Swish{
        convertDataType<CUDAPlugin::kernel::Type_t>(input_element_type), max_threads_per_block, num_elements, beta};
}

void SwishOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() >= 1);
    Expects(outputTensors.size() == 1);
    const auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(), inputTensors[0].get(), outputTensors[0].get());
}

OPERATION_REGISTER(SwishOp, Swish);
}  // namespace CUDAPlugin
