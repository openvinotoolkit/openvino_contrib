// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <ngraph/op/convolution.hpp>
#include <ngraph/type/element_type.hpp>

#include "cuda_plugin_custom_node_types.hpp"
#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"

namespace CUDAPlugin::nodes {

class FusedConvolution : public ngraph::op::Op {
public:
    explicit FusedConvolution(const ngraph::Output<Node>& data_batch,
                              const ngraph::Output<Node>& filters,
                              const ngraph::Output<Node>& bias,
                              const ngraph::Strides& strides,
                              const ngraph::CoordinateDiff& pads_begin,
                              const ngraph::CoordinateDiff& pads_end,
                              const ngraph::Strides& dilations,
                              const ngraph::op::PadType& auto_pad,
                              ActivationMode activation);
    explicit FusedConvolution(const ngraph::Output<Node>& data_batch,
                              const ngraph::Output<Node>& filters,
                              const ngraph::Output<Node>& bias,
                              const ngraph::Output<Node>& addArgNode,
                              const ngraph::Strides& strides,
                              const ngraph::CoordinateDiff& pads_begin,
                              const ngraph::CoordinateDiff& pads_end,
                              const ngraph::Strides& dilations,
                              const ngraph::op::PadType& auto_pad,
                              ActivationMode activation);

    inline static constexpr type_info_t type_info{"FusedConvolution", 0};
    const type_info_t& get_type_info() const override { return type_info; }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    void validate_and_infer_types() override;
    void conv_validate_and_infer_types();

    bool has_add_node() const;

    void set_activation(ActivationMode act);
    ActivationMode get_activation() const;

    const auto& get_strides() const { return strides_; }
    const auto& get_dilations() const { return dilations_; }
    const auto& get_pads_begin() const { return pads_begin_; }
    const auto& get_pads_end() const { return pads_end_; }
    const auto& get_auto_pad() const { return auto_pad_; }

private:
    /// Used for the shape validation
    ngraph::Strides strides_;
    ngraph::CoordinateDiff pads_begin_;
    ngraph::CoordinateDiff pads_end_;
    ngraph::Strides dilations_;
    ngraph::op::PadType auto_pad_;
    ngraph::Shape bias_shape_;
    ngraph::element::Type bias_type_;

    bool has_add_node_ = false;
    ActivationMode activation_;
};

}  // namespace CUDAPlugin::nodes
