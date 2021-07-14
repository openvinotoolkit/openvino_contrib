// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <ngraph/op/convolution.hpp>

#include "transformer/nodes/fused_convolution2d.hpp"

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

    size_t NumberOfDims() const { return input_shape_.size(); }
    size_t NumberOfSpatialDims() const { return input_shape_.size() - NON_SPATIAL_DIMS_NUMBER; }

private:
    template <typename TConvNode>
    void InferPadding(const TConvNode& node);
    void ConvertConv1DToConv2D();
};


/**
 * @brief Defines tensor indices for the following nodes:
 *  - `CUDAPlugin::nodes::FusedConv2D`
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
 *  - `CUDAPlugin::nodes::FusedConv2D`
 */
struct FusedConvolutionParams {
  FusedConvolutionParams(const CUDAPlugin::nodes::FusedConv2D& node);

    ConvolutionParams conv_;
    ngraph::Shape bias_shape_;
    std::optional<ngraph::Shape> add_shape_;
    CUDAPlugin::nodes::ActivationMode activation_;
};

} // namespace CUDAPlugin::Convolution::Details
