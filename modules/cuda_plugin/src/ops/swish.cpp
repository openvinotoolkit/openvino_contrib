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
double beta_from_constant(const ngraph::Node& swishNode) {
    constexpr auto tensorIndex = 1;
    constexpr auto defaultValue = 1.0;
    if (tensorIndex >= swishNode.get_input_size()) {
        return defaultValue;
    }
    const ngraph::Node* node = swishNode.get_input_node_ptr(tensorIndex);
    const ngraph::op::v0::Constant* constant = dynamic_cast<const ngraph::op::v0::Constant*>(node);
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
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)), beta_{beta_from_constant(node)} {
    Expects(node.get_input_size() == 1 || node.get_input_size() == 2);
    Expects(node.get_output_size() == 1);
    auto input_element_type = node.get_input_element_type(0);
    auto output_element_type = node.get_output_element_type(0);
    Expects(input_element_type == output_element_type);
    auto input_shape = node.get_input_shape(0);
    auto output_shape = node.get_output_shape(0);
    Expects(input_shape == output_shape);
    num_elements_ = ngraph::shape_size(input_shape);
    const size_t max_threads_per_block = context.device().props().maxThreadsPerBlock;
    kernel_ = kernel::Swish{convertDataType<CUDAPlugin::kernel::Type_t>(input_element_type), max_threads_per_block};
}

void SwishOp::Execute(const InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const {
    Expects(kernel_);
    Expects(inputTensors.size() >= 1);
    Expects(outputTensors.size() == 1);
    auto& stream = context.getThreadContext().stream();
    (*kernel_)(stream.get(),
               static_cast<const void*>(inputTensors[0].get()),
               static_cast<void*>(outputTensors[0].get()),
               num_elements_,
               beta_);
}

OPERATION_REGISTER(SwishOp, Swish);
}  // namespace CUDAPlugin
