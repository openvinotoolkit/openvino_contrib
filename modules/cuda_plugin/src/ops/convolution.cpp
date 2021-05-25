// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>

#include "convolution.hpp"

#include <sstream>
#include <fmt/format.h>

#include <ngraph/validation_util.hpp>

#include "cuda_operation_registry.hpp"
#include "convolution_cudnn.hpp"

namespace CUDAPlugin {

constexpr int NOT_SPATIAL_DIMS_NUMBER = 2;
constexpr int CONV_1D_DIMS_NUMBER = NOT_SPATIAL_DIMS_NUMBER + 1;

ConvolutionOp::ConvolutionOp(const NodeOp& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : OperationBase(node, std::move(inputIds), std::move(outputIds)) {
    const auto element_type = node.get_input_element_type(ArgIndices::input);
    Expects(element_type == node.get_input_element_type(ArgIndices::filter));
    Expects(element_type == node.get_output_element_type(ArgIndices::output));

    const auto& input_shape = node.get_input_shape(ArgIndices::input);
    const auto& filter_shape = node.get_input_shape(ArgIndices::filter);
    const auto& output_shape = node.get_output_shape(ArgIndices::output);
    const auto& strides = node.get_strides();
    const auto& dilations = node.get_dilations();
    const auto padding = InferPadding(node);

    if (input_shape.size() == CONV_1D_DIMS_NUMBER) {
        Create1DImpl(element_type, input_shape, filter_shape, output_shape,
                     strides, dilations, padding);
    } else {
        Create2D3DImpl(element_type, input_shape, filter_shape, output_shape,
                       strides, dilations, padding);
    }
}

void ConvolutionOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
    impl_->Execute(context, inputs, outputs);
}

ConvolutionOp::PaddingBeforeAndAfter
ConvolutionOp::InferPadding(const ngraph::op::v1::Convolution& op) {
    const ngraph::Shape& input_shape = op.get_input_shape(ArgIndices::input);
    const ngraph::op::PadType pad_type = op.get_auto_pad();
    switch (pad_type) {
    case ngraph::op::PadType::SAME_LOWER:
    case ngraph::op::PadType::SAME_UPPER:
        {
            PaddingBeforeAndAfter padding {};
            const ngraph::Shape& filter_shape = op.get_input_shape(ArgIndices::filter);
            const ngraph::Shape filter_spatial_shape {filter_shape.begin() + NOT_SPATIAL_DIMS_NUMBER,
                                                      filter_shape.end()};
            ngraph::infer_auto_padding(input_shape, filter_spatial_shape,
                                       op.get_strides(), op.get_dilations(), pad_type,
                                       padding.first, padding.second);
            return padding;
        } break;
    case ngraph::op::PadType::EXPLICIT:
        return std::make_pair(op.get_pads_begin(), op.get_pads_end());
    case ngraph::op::PadType::VALID:
        {
            size_t spatial_dims_number = input_shape.size() - NOT_SPATIAL_DIMS_NUMBER;
            return std::make_pair(ngraph::CoordinateDiff(spatial_dims_number, 0),
                                  ngraph::CoordinateDiff(spatial_dims_number, 0));
        }
    default:
        Expects(false);
    }
}

void ConvolutionOp::Create1DImpl(ngraph::element::Type_t element_type,
                                 ngraph::Shape input_shape,
                                 ngraph::Shape filter_shape,
                                 ngraph::Shape output_shape,
                                 ngraph::Strides strides,
                                 ngraph::Strides dilations,
                                 PaddingBeforeAndAfter padding) {
    // Turn 1D Convolution into 2D.
    Expects(input_shape.size() == CONV_1D_DIMS_NUMBER);
    input_shape.insert(input_shape.begin() + NOT_SPATIAL_DIMS_NUMBER, 1);
    Expects(filter_shape.size() == CONV_1D_DIMS_NUMBER);
    filter_shape.insert(filter_shape.begin() + NOT_SPATIAL_DIMS_NUMBER, 1);
    Expects(output_shape.size() == CONV_1D_DIMS_NUMBER);
    output_shape.insert(output_shape.begin() + NOT_SPATIAL_DIMS_NUMBER, 1);
    strides.insert(strides.begin(), 1);
    dilations.insert(dilations.begin(), 1);
    padding.first.insert(padding.first.begin(), 0);
    padding.second.insert(padding.second.begin(), 0);
    Create2D3DImpl(element_type, input_shape, filter_shape, output_shape,
                   strides, dilations, padding);
}

void ConvolutionOp::Create2D3DImpl(ngraph::element::Type_t element_type,
                                   const ngraph::Shape& input_shape,
                                   const ngraph::Shape& filter_shape,
                                   const ngraph::Shape& output_shape,
                                   const ngraph::Strides& strides,
                                   const ngraph::Strides& dilations,
                                   const PaddingBeforeAndAfter& padding) {
    const size_t dims_number = input_shape.size();
    const size_t spatial_dims_number = dims_number - NOT_SPATIAL_DIMS_NUMBER;
    Expects(input_shape.size() == dims_number);
    Expects(filter_shape.size() == dims_number);
    Expects(output_shape.size() == dims_number);
    Expects(strides.size() == spatial_dims_number);
    Expects(dilations.size() == spatial_dims_number);
    Expects(padding.first.size() == spatial_dims_number);
    Expects(padding.second.size() == spatial_dims_number);

    std::stringstream exception_msg;

    try {
        impl_ = std::make_unique<ConvolutionCuDnn>(
                    element_type, input_shape, filter_shape, output_shape,
                    strides, dilations, padding.first, padding.second);
        return;
    } catch(const std::exception& e) {
        exception_msg << "Failed to create ConvolutionCuDnn impl: " << e.what() << std::endl;
    }

    THROW_IE_EXCEPTION << fmt::format("Convolution node is not supported:\n {}", exception_msg.str());
}

OPERATION_REGISTER(ConvolutionOp, Convolution);
} // namespace CUDAPlugin
