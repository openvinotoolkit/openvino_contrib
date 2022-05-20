// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/group_conv.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <type_traits>

#include "cuda_plugin_custom_node_types.hpp"

namespace CUDAPlugin::nodes {

namespace {

template <typename TBaseConvolution>
inline constexpr const char* conv_name;

template <>
inline constexpr const char* conv_name<ngraph::op::v1::Convolution> = "FusedConvolution";

template <>
inline constexpr const char* conv_name<ngraph::op::v1::GroupConvolution> = "FusedGroupConvolution";

}  // namespace

template <typename TBaseConvolution>
class BasicFusedConvolution : public TBaseConvolution {
    static_assert(std::is_same_v<TBaseConvolution, ngraph::op::v1::Convolution> ||
                      std::is_same_v<TBaseConvolution, ngraph::op::v1::GroupConvolution>,
                  "TBaseConvolution should be either ngraph::op::v1::Convolution or ngraph::op::v1::GroupConvolution");

public:
    using BaseConvolution = TBaseConvolution;

    explicit BasicFusedConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                                   const ngraph::Output<ngraph::Node>& filters,
                                   const ngraph::Output<ngraph::Node>& bias,
                                   const ngraph::Strides& strides,
                                   const ngraph::CoordinateDiff& pads_begin,
                                   const ngraph::CoordinateDiff& pads_end,
                                   const ngraph::Strides& dilations,
                                   const ngraph::op::PadType& auto_pad,
                                   ActivationMode activation)
        : TBaseConvolution(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad),
          bias_shape_{bias.get_shape()},
          bias_type_{bias.get_element_type()},
          has_add_node_{false},
          activation_{activation} {
        TBaseConvolution::set_arguments({bias});
        TBaseConvolution::constructor_validate_and_infer_types();
    }

    explicit BasicFusedConvolution(const ngraph::Output<ngraph::Node>& data_batch,
                                   const ngraph::Output<ngraph::Node>& filters,
                                   const ngraph::Output<ngraph::Node>& bias,
                                   const ngraph::Output<ngraph::Node>& add_node,
                                   const ngraph::Strides& strides,
                                   const ngraph::CoordinateDiff& pads_begin,
                                   const ngraph::CoordinateDiff& pads_end,
                                   const ngraph::Strides& dilations,
                                   const ngraph::op::PadType& auto_pad,
                                   ActivationMode activation)
        : TBaseConvolution(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad),
          bias_shape_{bias.get_shape()},
          bias_type_{bias.get_element_type()},
          has_add_node_{true},
          activation_{activation} {
        TBaseConvolution::set_arguments({bias, add_node});
        TBaseConvolution::constructor_validate_and_infer_types();
    }

    inline static constexpr ngraph::Node::type_info_t type_info{conv_name<TBaseConvolution>, 0};

    const ngraph::Node::type_info_t& get_type_info() const override { return type_info; }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        ngraph::check_new_args_count(this, new_args);
        if (new_args.size() == 3) {
            return std::make_shared<BasicFusedConvolution<TBaseConvolution>>(new_args.at(0),
                                                                             new_args.at(1),
                                                                             new_args.at(2),
                                                                             TBaseConvolution::m_strides,
                                                                             TBaseConvolution::m_pads_begin,
                                                                             TBaseConvolution::m_pads_end,
                                                                             TBaseConvolution::m_dilations,
                                                                             TBaseConvolution::m_auto_pad,
                                                                             activation_);
        } else {
            return std::make_shared<BasicFusedConvolution<TBaseConvolution>>(new_args.at(0),
                                                                             new_args.at(1),
                                                                             new_args.at(2),
                                                                             new_args.at(3),
                                                                             TBaseConvolution::m_strides,
                                                                             TBaseConvolution::m_pads_begin,
                                                                             TBaseConvolution::m_pads_end,
                                                                             TBaseConvolution::m_dilations,
                                                                             TBaseConvolution::m_auto_pad,
                                                                             activation_);
        }
    }

    void validate_and_infer_types() override {
        TBaseConvolution::validate_and_infer_types();
        const auto& conv_out_shape = TBaseConvolution::get_output_shape(0);
        const auto& element_type = TBaseConvolution::get_output_element_type(0);

        const size_t num_spatial_dims = conv_out_shape.size() - 2;
        const size_t nchw_channel_dim_offset = num_spatial_dims + 1;
        constexpr size_t conv_output_rank_max{5};
        constexpr size_t conv_bias_rank_min{3};
        Expects(conv_out_shape.size() <= conv_output_rank_max);
        Expects(bias_shape_.size() >= conv_bias_rank_min);
        const size_t conv_channel_dim_size = conv_out_shape.at(conv_out_shape.size() - nchw_channel_dim_offset);
        const size_t bias_channel_dim_size = bias_shape_.at(bias_shape_.size() - nchw_channel_dim_offset);

        Expects(conv_channel_dim_size == bias_channel_dim_size);
        Expects(bias_type_ == element_type);

        TBaseConvolution::set_output_type(0, element_type, conv_out_shape);
    }

    bool has_add_node() const { return has_add_node_; }

    void set_activation(ActivationMode mode) { activation_ = mode; }

    ActivationMode get_activation() const { return activation_; }

private:
    ngraph::Shape bias_shape_;
    ngraph::element::Type bias_type_;
    bool has_add_node_;
    ActivationMode activation_;
};  // class TBaseConvolution

class FusedConvolution : public BasicFusedConvolution<ngraph::op::v1::Convolution> {
public:
    using BasicFusedConvolution::BasicFusedConvolution;
};
class FusedGroupConvolution : public BasicFusedConvolution<ngraph::op::v1::GroupConvolution> {
public:
    using BasicFusedConvolution::BasicFusedConvolution;
};

}  // namespace CUDAPlugin::nodes
