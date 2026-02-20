// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "activation_type.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov::nvidia_gpu::nodes {

namespace {

template <typename TBaseConvolution>
inline constexpr const char* conv_name;

template <>
inline constexpr const char* conv_name<ov::op::v1::Convolution> = "FusedConvolution";

template <>
inline constexpr const char* conv_name<ov::op::v1::GroupConvolution> = "FusedGroupConvolution";

}  // namespace

template <typename TBaseConvolution>
class BasicFusedConvolution : public TBaseConvolution {
    static_assert(std::is_same_v<TBaseConvolution, ov::op::v1::Convolution> ||
                      std::is_same_v<TBaseConvolution, ov::op::v1::GroupConvolution>,
                  "TBaseConvolution should be either ov::op::v1::Convolution or ov::op::v1::GroupConvolution");

public:
    using BaseConvolution = TBaseConvolution;

    OPENVINO_OP(conv_name<TBaseConvolution>, "nvidia_gpu", TBaseConvolution);

    BasicFusedConvolution() = default;
    ~BasicFusedConvolution() = default;

    explicit BasicFusedConvolution(const ov::Output<ov::Node>& data_batch,
                                   const ov::Output<ov::Node>& filters,
                                   const ov::Output<ov::Node>& bias,
                                   const ov::Strides& strides,
                                   const ov::CoordinateDiff& pads_begin,
                                   const ov::CoordinateDiff& pads_end,
                                   const ov::Strides& dilations,
                                   const ov::op::PadType& auto_pad,
                                   ActivationMode activation)
        : TBaseConvolution(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad),
          activation_{activation} {
        TBaseConvolution::set_arguments({data_batch, filters, bias});
        TBaseConvolution::constructor_validate_and_infer_types();
    }

    explicit BasicFusedConvolution(const ov::Output<ov::Node>& data_batch,
                                   const ov::Output<ov::Node>& filters,
                                   const ov::Output<ov::Node>& bias,
                                   const ov::Output<ov::Node>& add_node,
                                   const ov::Strides& strides,
                                   const ov::CoordinateDiff& pads_begin,
                                   const ov::CoordinateDiff& pads_end,
                                   const ov::Strides& dilations,
                                   const ov::op::PadType& auto_pad,
                                   ActivationMode activation)
        : TBaseConvolution(data_batch, filters, strides, pads_begin, pads_end, dilations, auto_pad),
          activation_{activation} {
        TBaseConvolution::set_arguments({data_batch, filters, bias, add_node});
        TBaseConvolution::constructor_validate_and_infer_types();
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        ov::check_new_args_count(this, new_args);
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

    bool visit_attributes(AttributeVisitor& visitor) override {
        TBaseConvolution::visit_attributes(visitor);
        visitor.on_attribute("activation", activation_);
        return true;
    }

    void validate_and_infer_types() override {
        TBaseConvolution::validate_and_infer_types();
        OPENVINO_ASSERT(TBaseConvolution::get_input_size() == 3 || TBaseConvolution::get_input_size() == 4);
        const auto& conv_out_shape = TBaseConvolution::get_output_shape(0);
        const auto& element_type = TBaseConvolution::get_output_element_type(0);

        const size_t num_spatial_dims = conv_out_shape.size() - 2;
        const size_t nchw_channel_dim_offset = num_spatial_dims + 1;
        constexpr size_t conv_output_rank_max{5};
        constexpr size_t conv_bias_rank_min{3};
        OPENVINO_ASSERT(conv_out_shape.size() <= conv_output_rank_max);
        auto bias_shape = TBaseConvolution::get_input_shape(2);
        OPENVINO_ASSERT(bias_shape.size() >= conv_bias_rank_min);
        const size_t conv_channel_dim_size = conv_out_shape.at(conv_out_shape.size() - nchw_channel_dim_offset);
        const size_t bias_channel_dim_size = bias_shape.at(bias_shape.size() - nchw_channel_dim_offset);

        OPENVINO_ASSERT(conv_channel_dim_size == bias_channel_dim_size);
        OPENVINO_ASSERT(TBaseConvolution::get_input_element_type(2) == element_type);

        TBaseConvolution::set_output_type(0, element_type, conv_out_shape);
    }

    bool has_add_node() const { return (TBaseConvolution::get_input_size() == 4); }

    void set_activation(ActivationMode mode) { activation_ = mode; }

    ActivationMode get_activation() const { return activation_; }

private:
    ActivationMode activation_;
};  // class TBaseConvolution

using FusedConvolution = BasicFusedConvolution<ov::op::v1::Convolution>;
using FusedGroupConvolution = BasicFusedConvolution<ov::op::v1::GroupConvolution>;

}  // namespace ov::nvidia_gpu::nodes
