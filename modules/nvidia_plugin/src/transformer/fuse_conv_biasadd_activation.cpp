// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <type_traits>
#include <utility>
#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "fuse_conv_biasadd_activation.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"


#include "nodes/activation_type.hpp"
#include "nodes/fused_convolution.hpp"
#include "nodes/fused_convolution_backprop_data.hpp"
#include "rt_info/cuda_node_id.hpp"

using namespace ov::pass::pattern;

using ov::nvidia_gpu::nodes::FusedConvolution;
using ov::nvidia_gpu::nodes::FusedGroupConvolution;

using ActivationMode = ov::nvidia_gpu::nodes::ActivationMode;
using FusedConvBackpropData = ov::nvidia_gpu::nodes::FusedConvBackpropData;

using namespace ov::nvidia_gpu::rt_info;

namespace {
template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(std::shared_ptr<ov::Node> node) {
    auto eltwise = std::dynamic_pointer_cast<A>(node->input(0).get_source_output().get_node_shared_ptr());
    auto constant = std::dynamic_pointer_cast<B>(node->input(1).get_source_output().get_node_shared_ptr());

    if (!eltwise) {
        eltwise = std::dynamic_pointer_cast<A>(node->input(1).get_source_output().get_node_shared_ptr());
        constant = std::dynamic_pointer_cast<B>(node->input(0).get_source_output().get_node_shared_ptr());
    }

    if (!eltwise || !constant) {
        return {nullptr, nullptr};
    }

    return {eltwise, constant};
}

template <typename TFusedConvolution>
struct FusedConvCallbacks {
    static_assert(std::is_same_v<TFusedConvolution, FusedConvolution> ||
                      std::is_same_v<TFusedConvolution, FusedGroupConvolution>,
                  "TFusedConvolution should be either FusedConvolution or FusedGroupConvolution");
    static bool fuse_convolution_with_biasadd(Matcher &m) {
        auto eltwise = m.get_match_root();
        auto [m_conv, m_const] =
            parse_eltwise_inputs<typename TFusedConvolution::BaseConvolution, ov::op::v0::Constant>(eltwise);
        if (!m_conv || !m_const) {
            return false;
        }

        if (m_conv->inputs().size() != 2) {
            return false;
        }

        if (std::dynamic_pointer_cast<ov::op::v1::Add>(eltwise) == nullptr) {
            return false;
        }

        const ov::Output<ov::Node> &data = m_conv->input(0).get_source_output();
        const ov::Output<ov::Node> &filters = m_conv->input(1).get_source_output();
        const ov::Output<ov::Node> &bias = m_const->output(0);

        auto fused_conv = std::make_shared<TFusedConvolution>(data,
                                                              filters,
                                                              bias,
                                                              m_conv->get_strides(),
                                                              m_conv->get_pads_begin(),
                                                              m_conv->get_pads_end(),
                                                              m_conv->get_dilations(),
                                                              m_conv->get_auto_pad(),
                                                              ActivationMode::NO_ACTIVATION);
        ov::Output<ov::Node> new_conv(fused_conv);

        fused_conv->set_friendly_name(eltwise->get_friendly_name());

        ov::copy_runtime_info({m_conv, eltwise}, new_conv.get_node_shared_ptr());
        set_node_id(new_conv.get_node_shared_ptr(), get_node_id(eltwise));

        ov::replace_node(m.get_match_root(), new_conv.get_node_shared_ptr());
        return true;
    }

    static std::pair<std::shared_ptr<TFusedConvolution>, std::shared_ptr<ov::Node>> parse_fusedconv_inputs(
        std::shared_ptr<ov::Node> add) {
        std::shared_ptr<TFusedConvolution> fused_conv = nullptr;

        auto input0 = add->input(0).get_source_output().get_node_shared_ptr();
        auto input1 = add->input(1).get_source_output().get_node_shared_ptr();

        auto fused_conv0 = std::dynamic_pointer_cast<TFusedConvolution>(input0);
        auto fused_conv1 = std::dynamic_pointer_cast<TFusedConvolution>(input1);

        auto can_be_fused = [](const std::shared_ptr<ov::Node>& target, const std::shared_ptr<ov::Node>& fused_input) {
            return (target && fused_input && (get_node_id(target) > get_node_id(fused_input) || ov::op::util::is_constant(fused_input)));
        };

        if (fused_conv0 && fused_conv1) {
            if (can_be_fused(fused_conv0, input1)) {
                return {fused_conv0, input1};
            } else if (can_be_fused(fused_conv1, input0)) {
                return {fused_conv1, input0};
            }
        }

        if (fused_conv0 && can_be_fused(fused_conv0, input1)) {
            return {fused_conv0, input1};
        }

        if (fused_conv1 && can_be_fused(fused_conv1, input0)) {
            return {fused_conv1, input0};
        }
        return {nullptr, nullptr};
    }

    static bool sink_add_to_fused_convolution(Matcher &m) {
        auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
        auto [fused_conv, node] = parse_fusedconv_inputs(m.get_match_root());
        if (!fused_conv || !node) {
            return false;
        }

        if (fused_conv->has_add_node() || fused_conv->get_activation() != ActivationMode::NO_ACTIVATION) {
            return false;
        }

        const ov::Output<ov::Node> &data = fused_conv->input(0).get_source_output();
        const ov::Output<ov::Node> &filters = fused_conv->input(1).get_source_output();
        const ov::Output<ov::Node> &bias = fused_conv->input(2).get_source_output();

        auto fused_conv_add = std::make_shared<TFusedConvolution>(data,
                                                                  filters,
                                                                  bias,
                                                                  node,
                                                                  fused_conv->get_strides(),
                                                                  fused_conv->get_pads_begin(),
                                                                  fused_conv->get_pads_end(),
                                                                  fused_conv->get_dilations(),
                                                                  fused_conv->get_auto_pad(),
                                                                  ActivationMode::NO_ACTIVATION);
        ov::Output<ov::Node> fused_conv_add_output{fused_conv_add};

        fused_conv_add->set_friendly_name(add->get_friendly_name());
        ov::copy_runtime_info({node, fused_conv}, fused_conv_add);
        set_node_id(fused_conv_add, get_node_id(add));

        ov::replace_node(fused_conv, fused_conv_add);
        ov::replace_node(m.get_match_root(), fused_conv_add);

        return true;
    }

    static bool sink_activation_to_fused_convolution(Matcher &m) {
        auto activationNode = m.get_match_root();
        auto fused_conv = std::dynamic_pointer_cast<TFusedConvolution>(
            activationNode->input(0).get_source_output().get_node_shared_ptr());
        if (fused_conv->get_activation() != ActivationMode::NO_ACTIVATION) {
            return false;
        }

        ActivationMode activation = ActivationMode::NO_ACTIVATION;
        if (ov::is_type<ov::op::v0::Relu>(activationNode)) {
            activation = ActivationMode::RELU;
        } else if (ov::is_type<ov::op::v0::Sigmoid>(activationNode)) {
            activation = ActivationMode::SIGMOID;
        } else if (ov::is_type<ov::op::v0::Tanh>(activationNode)) {
            activation = ActivationMode::TANH;
        } else {
            return false;
        }
        fused_conv->set_activation(activation);

        fused_conv->set_friendly_name(activationNode->get_friendly_name());
        set_node_id(fused_conv, get_node_id(activationNode));

        ov::replace_node(m.get_match_root(), fused_conv);

        return true;
    }
};  // struct FusedConvCallbacks

bool fuse_convolution_backprop_data_with_add(Matcher &m) {
    auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(m.get_match_root());
    auto [conv_backprop_data, add_constant] =
        parse_eltwise_inputs<ov::op::v1::ConvolutionBackpropData, ov::op::v0::Constant>(add);

    const auto conv_element_type = conv_backprop_data->get_input_element_type(0);
    const auto add_element_type = add_constant->get_output_element_type(0);
    if (conv_element_type != add_element_type) {
        return false;
    }

    const auto conv_output_shape = dynamic_cast<ov::Node *>(conv_backprop_data.get())->get_output_shape(0);
    const auto add_output_shape = add_constant->get_output_shape(0);
    const auto size = ov::element::Type(conv_element_type).size();
    const auto conv_in_bytes = size * ov::shape_size(conv_output_shape);
    const auto add_in_bytes = size * ov::shape_size(add_output_shape);
    if (add_in_bytes > conv_in_bytes) {
        return false;
    }

    const ov::Output<ov::Node> &data = conv_backprop_data->input(0).get_source_output();
    const ov::Output<ov::Node> &filters = conv_backprop_data->input(1).get_source_output();
    std::shared_ptr<FusedConvBackpropData> fused_conv_backprop_data_add;

    if (3 == conv_backprop_data->get_input_size()) {
        auto output_shape = conv_backprop_data->input(2).get_source_output();
        fused_conv_backprop_data_add =
            std::make_shared<FusedConvBackpropData>(data,
                                                    filters,
                                                    output_shape,
                                                    add_constant,
                                                    conv_backprop_data->get_strides(),
                                                    conv_backprop_data->get_pads_begin(),
                                                    conv_backprop_data->get_pads_end(),
                                                    conv_backprop_data->get_dilations(),
                                                    conv_backprop_data->get_auto_pad(),
                                                    conv_backprop_data->get_output_padding());
    } else {
        fused_conv_backprop_data_add =
            std::make_shared<FusedConvBackpropData>(data,
                                                    filters,
                                                    add_constant,
                                                    conv_backprop_data->get_strides(),
                                                    conv_backprop_data->get_pads_begin(),
                                                    conv_backprop_data->get_pads_end(),
                                                    conv_backprop_data->get_dilations(),
                                                    conv_backprop_data->get_auto_pad(),
                                                    conv_backprop_data->get_output_padding());
    }

    ov::Output<ov::Node> fused_conv_backprop_data_add_output{fused_conv_backprop_data_add};

    fused_conv_backprop_data_add->set_friendly_name(add->get_friendly_name());
    ov::copy_runtime_info({conv_backprop_data, add}, fused_conv_backprop_data_add);

    ov::replace_node(add, fused_conv_backprop_data_add);
    ov::replace_node(conv_backprop_data, fused_conv_backprop_data_add);

    return true;
}
bool is_bias_to_be_fused(const ov::Output<ov::Node>& output) {
    constexpr auto conv_bias_rank_min{3};
    constexpr auto conv_bias_rank_max{5};
    auto node = std::dynamic_pointer_cast<ov::op::v1::Add>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }

    auto input0 = node->input(0);
    auto input1 = node->input(1);

    const auto partial_shape0 = node->input(0).get_partial_shape();
    const auto partial_shape1 = node->input(1).get_partial_shape();

    if (partial_shape0.is_dynamic() || partial_shape1.is_dynamic()) {
        return false;
    }

    if (node->get_autob() != ov::op::AutoBroadcastType::NUMPY) {
        return false;
    }

    if (input0.get_element_type() != input1.get_element_type()) {
        return false;
    }

    const auto conv_shape = partial_shape0.to_shape();
    const auto bias_shape = partial_shape1.to_shape();
    const auto bias_rank = bias_shape.size();
    if (bias_rank < conv_bias_rank_min || bias_rank > conv_bias_rank_max) {
        return false;
    }
    const auto num_spatial_dims = conv_shape.size() - 2;
    if (num_spatial_dims == 3) {
        return false;  // NOTE: 3D convolution fusing was disabled due to 3d_unet bad performance
    }
    const auto nchw_channel_dim_reverse_offset = num_spatial_dims + 1;
    size_t bias_channel_index = bias_shape.size() - nchw_channel_dim_reverse_offset;
    size_t conv_channel_index = conv_shape.size() - nchw_channel_dim_reverse_offset;
    if (bias_shape.at(bias_channel_index) != conv_shape.at(conv_channel_index)) {
        return false;
    }
    for (size_t i = 0; i < bias_shape.size(); i++) {
        if ((i != bias_channel_index) && (bias_shape.at(i) != 1)) return false;
    }
    return true;
}
bool is_add_to_be_fused(const ov::Output<ov::Node>& output) {
    auto node = std::dynamic_pointer_cast<ov::op::v1::Add>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }

    auto input0 = node->input(0);
    auto input1 = node->input(1);

    const auto partial_shape0 = node->input(0).get_partial_shape();
    const auto partial_shape1 = node->input(1).get_partial_shape();

    if (input0.get_element_type() != input1.get_element_type()) {
        return false;
    }

    if (partial_shape0.is_dynamic() || partial_shape1.is_dynamic()) {
        return false;
    }
    return (partial_shape0.to_shape() == partial_shape1.to_shape());
}
} // namespace

bool ov::nvidia_gpu::pass::CudaFuseMarkUpNodesOrder::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(CudaFuseMarkUpNodesOrder);
    uint64_t id = 0;
    for (auto& node : m->get_ordered_ops()) {
        set_node_id(node, id++);
    }
    return false;
}

bool ov::nvidia_gpu::pass::CudaFuseCleanUpNodesOrder::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(CudaFuseCleanUpNodesOrder);
    for (auto& node : m->get_ordered_ops()) {
        remove_node_id(node);
    }
    return false;
}

ov::nvidia_gpu::pass::FuseConvolutionWithBiasAdd::FuseConvolutionWithBiasAdd() {
    MATCHER_SCOPE(FuseConvolutionWithBiasAdd);
    auto conv = wrap_type<ov::op::v1::Convolution>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({conv, bias}, is_bias_to_be_fused);

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedConvolution>::fuse_convolution_with_biasadd(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::nvidia_gpu::pass::FuseGroupConvolutionWithBiasAdd::FuseGroupConvolutionWithBiasAdd() {
    MATCHER_SCOPE(FuseGroupConvolutionWithBiasAdd);
    auto conv = wrap_type<ov::op::v1::GroupConvolution>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({conv, bias}, is_bias_to_be_fused);

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedGroupConvolution>::fuse_convolution_with_biasadd(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::nvidia_gpu::pass::FuseConvolutionWithBiasAddAdd::FuseConvolutionWithBiasAddAdd() {
    MATCHER_SCOPE(FuseConvolutionWithBiasAddAdd);
    auto fused_convolution = wrap_type<FusedConvolution>(consumers_count(1));
    auto add1 = wrap_type<ov::op::v1::Add>({fused_convolution, any_input()}, is_add_to_be_fused);
    auto add2 = wrap_type<ov::op::v1::Add>({any_input(), fused_convolution}, is_add_to_be_fused);
    auto add = std::make_shared<::op::Or>(ov::OutputVector{ add1, add2 });

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedConvolution>::sink_add_to_fused_convolution(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::nvidia_gpu::pass::FuseGroupConvolutionWithBiasAddAdd::FuseGroupConvolutionWithBiasAddAdd() {
    MATCHER_SCOPE(FuseGroupConvolutionWithBiasAddAdd);
    auto fused_convolution = wrap_type<FusedGroupConvolution>(consumers_count(1));
    auto add1 = wrap_type<ov::op::v1::Add>({fused_convolution, any_input()}, is_add_to_be_fused);
    auto add2 = wrap_type<ov::op::v1::Add>({any_input(), fused_convolution}, is_add_to_be_fused);
    auto add = std::make_shared<::op::Or>(ov::OutputVector{ add1, add2 });

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedGroupConvolution>::sink_add_to_fused_convolution(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}

ov::nvidia_gpu::pass::SinkActivationToFusedConvolution::SinkActivationToFusedConvolution() {
    MATCHER_SCOPE(SinkActivationToFusedConvolution);
    auto fused_convolution = wrap_type<FusedConvolution>(consumers_count(1));
    // TODO: Uncomment when performance for FusedConvolution+BiasAdd+Tanh would be satisfied
    // auto activation = wrap_type<ov::op::v0::Relu, ov::op::v0::Sigmoid, ov::op::v0::Tanh>({fused_convolution});
    auto activation = wrap_type<ov::op::v0::Relu, ov::op::v0::Sigmoid>({fused_convolution});

    matcher_pass_callback callback = [](Matcher &m) {
        return FusedConvCallbacks<FusedConvolution>::sink_activation_to_fused_convolution(m);
    };

    auto m = std::make_shared<Matcher>(activation, matcher_name);
    register_matcher(m, callback);
}

bool ov::nvidia_gpu::pass::CudaConvolutionFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(CudaConvolutionFusion);
    ov::pass::Manager manager(get_pass_config());

    manager.register_pass<CudaFuseMarkUpNodesOrder>();

    auto fuse_conv_bias_add_activation = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseConvolutionWithBiasAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, FuseConvolutionWithBiasAddAdd)
    ADD_MATCHER(fuse_conv_bias_add_activation, SinkActivationToFusedConvolution)
    fuse_conv_bias_add_activation->set_name("ov::nvidia_gpu::pass::fuse_conv_bias_add_activation");

    auto fuse_group_conv_bias_add_activation = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(fuse_group_conv_bias_add_activation, FuseGroupConvolutionWithBiasAdd)
    ADD_MATCHER(fuse_group_conv_bias_add_activation, FuseGroupConvolutionWithBiasAddAdd)
    // TODO: Activations, should check performance first
    fuse_group_conv_bias_add_activation->set_name("ov::nvidia_gpu::pass::fuse_group_conv_bias_add_activation");

    manager.register_pass<CudaFuseConvBackpropDataAdd>();
    manager.register_pass<CudaFuseCleanUpNodesOrder>();

    manager.run_passes(m);
    return false;
}

ov::nvidia_gpu::pass::CudaFuseConvBackpropDataAdd::CudaFuseConvBackpropDataAdd() {
    MATCHER_SCOPE(CudaFuseConvBackpropDataAdd);
    auto conv_backprop_data =
        wrap_type<ov::op::v1::ConvolutionBackpropData>(consumers_count(1));
    auto bias = wrap_type<ov::op::v0::Constant>();
    auto add = wrap_type<ov::op::v1::Add>({conv_backprop_data, bias}, is_add_to_be_fused);

    matcher_pass_callback callback = [](Matcher &m) {
        return fuse_convolution_backprop_data_with_add(m);
    };

    auto m = std::make_shared<Matcher>(add, matcher_name);
    register_matcher(m, callback);
}
