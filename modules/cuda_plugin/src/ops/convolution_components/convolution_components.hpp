// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/convolution.hpp>
#include <optional>

#include "transformer/nodes/fused_convolution.hpp"
#include "transformer/nodes/fused_convolution_backprop_data.hpp"

namespace CUDAPlugin::Convolution::Details {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;

/**
 * @brief Defines tensor indices for `ngraph::op::v1::Convolution` node.
 */
struct ConvArgIndices {
    static constexpr size_t input = 0;
    static constexpr size_t filter = 1;
    static constexpr size_t output = 0;
};

/**
 * @brief Unified convolution parameters as they are consumed by different
 * implementations.
 *
 * This class performs the following common tasks:
 *  - Extracts and validates required parameter values from ngraph operation;
 *  - Converts 1D convolution to 2D convolution;
 *  - Eliminates `ngraph::op::PadType` providing actual padding values.
 */
struct ConvolutionParams {
    template <typename TConvNode>
    ConvolutionParams(const TConvNode& node);

    ngraph::element::Type_t element_type_;
    ngraph::Shape input_shape_;
    ngraph::Shape filter_shape_;
    ngraph::Shape output_shape_;
    ngraph::Strides strides_;
    ngraph::Strides dilations_;
    ngraph::CoordinateDiff padding_before_;
    ngraph::CoordinateDiff padding_after_;
    size_t groups_;

    size_t NumberOfDims() const { return input_shape_.size(); }
    size_t NumberOfSpatialDims() const { return input_shape_.size() - NON_SPATIAL_DIMS_NUMBER; }

private:
    template <typename TConvNode>
    void InferPadding(const TConvNode& node);
    void ConvertConv1DToConv2D();
};

struct ConvBackArgIndices {
    static constexpr size_t doutput = 0;
    static constexpr size_t filter = 1;
    static constexpr size_t output_shape = 2;
    static constexpr size_t dinput = 0;
};

struct ConvolutionBackwardDataParams {
    template <typename TConvNode>
    ConvolutionBackwardDataParams(const TConvNode& node);

    ngraph::element::Type_t element_type_;
    ngraph::Shape doutput_shape_;
    ngraph::Shape filter_shape_;
    ngraph::Shape dinput_shape_;
    ngraph::Strides strides_;
    ngraph::Strides dilations_;
    ngraph::CoordinateDiff pads_begin_;
    ngraph::CoordinateDiff pads_end_;
    ngraph::op::PadType auto_pad_;
    ngraph::CoordinateDiff output_padding_;

    size_t NumberOfDims() const { return doutput_shape_.size(); }
    size_t NumberOfSpatialDims() const { return doutput_shape_.size() - NON_SPATIAL_DIMS_NUMBER; }

private:
    void ConvertConv1DToConv2D();
};

/**
 * @brief Defines tensor indices for the following nodes:
 *  - `CUDAPlugin::nodes::FusedConvolution`
 */
struct FusedConvolutionIndices {
    static constexpr size_t input = 0;
    static constexpr size_t filter = 1;
    static constexpr size_t bias = 2;
    static constexpr size_t add = 3;
    static constexpr size_t output = 0;
};

/**
 * @brief Unified parameters as they are consumed by the following nodes:
 *  - `CUDAPlugin::nodes::FusedConvolution`
 */
struct FusedConvolutionParams {
    template <typename TConvNode>
    FusedConvolutionParams(const TConvNode& node);

    ConvolutionParams conv_;
    ngraph::Shape bias_shape_;
    std::optional<ngraph::Shape> add_shape_;
    CUDAPlugin::nodes::ActivationMode activation_;
};

/**
 * @brief Defines tensor indices for the following nodes:
 *  - `CUDAPlugin::nodes::FusedConvBackpropData`
 */
template <std::size_t InputSize>
struct FusedConvolutionBackwardDataIndices;

template <>
struct FusedConvolutionBackwardDataIndices<3> {
    static constexpr size_t doutput = 0;
    static constexpr size_t filter = 1;
    static constexpr size_t add = 2;
    static constexpr size_t dinput = 0;
};

template <>
struct FusedConvolutionBackwardDataIndices<4> {
    static constexpr size_t doutput = 0;
    static constexpr size_t filter = 1;
    static constexpr size_t output_shape = 2;
    static constexpr size_t add = 3;
    static constexpr size_t dinput = 0;
};

/**
 * @brief Unified parameters as they are consumed by the following nodes:
 *  - `CUDAPlugin::nodes::FusedConvBackpropData`
 */
struct FusedConvolutionBackwardDataParams {
    FusedConvolutionBackwardDataParams(const CUDAPlugin::nodes::FusedConvBackpropData& node);

    ConvolutionBackwardDataParams conv_;
    ngraph::Shape add_shape_;
};

}  // namespace CUDAPlugin::Convolution::Details
