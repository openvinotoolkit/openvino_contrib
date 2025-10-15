// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/fuse_conv_biasadd_activation.hpp"

#include <gtest/gtest.h>

#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "transformer/nodes/fused_convolution.hpp"
#include "transformer/nodes/fused_convolution_backprop_data.hpp"

using ov::nvidia_gpu::nodes::FusedConvBackpropData;
using ov::nvidia_gpu::nodes::FusedConvolution;
using ov::nvidia_gpu::nodes::FusedGroupConvolution;
using ActivationMode = ov::nvidia_gpu::nodes::ActivationMode;
using namespace ov::opset10;

namespace testing {

namespace {

enum class ModelType {
    ConvBias = 0,
    ConvBiasAdd,
    ConvBiasMulAdd,
    ConvBiasConvBiasAdd,
    GroupConvBias,
    GroupConvBiasAdd,
    GroupConvBiasMulAdd,
    GroupConvBiasConvBiasAdd,
    ConvBackpropAdd
};

const std::vector<ModelType> conv_model_types = {
    ModelType::ConvBias, ModelType::ConvBiasAdd, ModelType::ConvBiasConvBiasAdd, ModelType::ConvBiasMulAdd};
const std::vector<ActivationMode> conv_activation_types = {
    ActivationMode::NO_ACTIVATION, ActivationMode::RELU, ActivationMode::SIGMOID};

const std::vector<ModelType> group_conv_model_types = {ModelType::GroupConvBias,
                                                       ModelType::GroupConvBiasAdd,
                                                       ModelType::GroupConvBiasConvBiasAdd,
                                                       ModelType::GroupConvBiasMulAdd};
const std::vector<ActivationMode> group_conv_activation_types = {ActivationMode::NO_ACTIVATION};

const std::vector<ModelType> conv_backprop_model_types = {ModelType::ConvBackpropAdd};
const std::vector<ActivationMode> conv_backprop_activation_types = {ActivationMode::NO_ACTIVATION};

struct ConvParams {
    ov::Shape input_shape;
    ov::Shape filter_shape;
    ov::Shape bias_shape;
    ov::Shape eltwise_shape;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    ov::op::PadType pad_type;
};

std::shared_ptr<ov::Model> createModel(const ModelType& model_type,
                                       const ActivationMode& act_type,
                                       const ConvParams& conv_params) {
    std::shared_ptr<ov::Node> last_op = nullptr;

    auto add_conv = [&](const ov::Output<ov::Node>& output) -> std::shared_ptr<Convolution> {
        return std::make_shared<Convolution>(output,
                                             Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                             conv_params.stride,
                                             conv_params.pads_begin,
                                             conv_params.pads_end,
                                             conv_params.dilation,
                                             conv_params.pad_type);
    };

    auto add_group_conv = [&](const ov::Output<ov::Node>& output) -> std::shared_ptr<GroupConvolution> {
        return std::make_shared<GroupConvolution>(output,
                                                  Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                                  conv_params.stride,
                                                  conv_params.pads_begin,
                                                  conv_params.pads_end,
                                                  conv_params.dilation,
                                                  conv_params.pad_type);
    };

    auto add_bias = [&](const ov::Output<ov::Node>& output) -> std::shared_ptr<Add> {
        return std::make_shared<Add>(output, Constant::create(ov::element::f32, conv_params.bias_shape, {1}));
    };

    auto add_eltwise = [&](const ov::Output<ov::Node>& output, bool multiply = false) -> std::shared_ptr<ov::Node> {
        if (multiply) {
            return std::make_shared<Multiply>(output,
                                              Constant::create(ov::element::f32, conv_params.eltwise_shape, {1}));
        } else {
            return std::make_shared<Add>(output, Constant::create(ov::element::f32, conv_params.eltwise_shape, {1}));
        }
    };

    auto add_act = [&](const ov::Output<ov::Node>& output) -> std::shared_ptr<ov::Node> {
        if (act_type == ActivationMode::RELU) {
            return std::make_shared<Relu>(output);
        } else if (act_type == ActivationMode::SIGMOID) {
            return std::make_shared<Sigmoid>(output);
        }
        return output.get_node_shared_ptr();
    };

    auto input_node = std::make_shared<Parameter>(ov::element::f32, conv_params.input_shape);
    if (model_type == ModelType::ConvBias) {
        auto conv = add_conv(input_node);
        auto bias = add_bias(conv);
        last_op = add_act(bias);
    } else if (model_type == ModelType::ConvBiasAdd) {
        auto conv = add_conv(input_node);
        auto bias = add_bias(conv);
        auto add = add_eltwise(bias);
        last_op = add_act(add);
    } else if (model_type == ModelType::ConvBiasMulAdd) {
        auto conv = add_conv(input_node);
        auto bias = add_bias(conv);
        last_op = add_act(bias);
        auto mul = add_eltwise(last_op, true);
        auto add = std::make_shared<Add>(last_op, mul);
        last_op = add;
    } else if (model_type == ModelType::ConvBiasConvBiasAdd) {
        auto conv = add_conv(input_node);
        auto bias = add_bias(conv);
        last_op = add_act(bias);
        auto conv2 = add_conv(last_op);
        auto bias2 = add_bias(conv2);
        auto add = std::make_shared<Add>(last_op, bias2);
        last_op = add;
    } else if (model_type == ModelType::GroupConvBias) {
        auto group_conv = add_group_conv(input_node);
        auto bias = add_bias(group_conv);
        last_op = add_act(bias);
    } else if (model_type == ModelType::GroupConvBiasAdd) {
        auto group_conv = add_group_conv(input_node);
        auto bias = add_bias(group_conv);
        auto add = add_eltwise(bias);
        last_op = add_act(add);
    } else if (model_type == ModelType::GroupConvBiasMulAdd) {
        auto group_conv = add_group_conv(input_node);
        auto bias = add_bias(group_conv);
        last_op = add_act(bias);
        auto mul = add_eltwise(last_op, true);
        auto add = std::make_shared<Add>(last_op, mul);
        last_op = add;
    } else if (model_type == ModelType::GroupConvBiasConvBiasAdd) {
        auto group_conv = add_group_conv(input_node);
        auto bias = add_bias(group_conv);
        last_op = add_act(bias);
        auto group_conv2 = add_group_conv(last_op);
        auto bias2 = add_bias(group_conv2);
        auto add = std::make_shared<Add>(last_op, bias2);
        last_op = add;
    }
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<Result>(last_op)},
                                       ov::ParameterVector{input_node});
}

template <typename TFusedConvolution>
std::shared_ptr<ov::Model> createRefModel(const ModelType& model_type,
                                          const ActivationMode& act_type,
                                          const ConvParams& conv_params) {
    std::shared_ptr<ov::Node> last_op = nullptr;
    auto input_node = std::make_shared<Parameter>(ov::element::f32, conv_params.input_shape);
    if (model_type == ModelType::ConvBias || model_type == ModelType::GroupConvBias) {
        auto fused_conv =
            std::make_shared<TFusedConvolution>(input_node,
                                                Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                                Constant::create(ov::element::f32, conv_params.bias_shape, {1}),
                                                conv_params.stride,
                                                conv_params.pads_begin,
                                                conv_params.pads_end,
                                                conv_params.dilation,
                                                conv_params.pad_type,
                                                act_type);
        last_op = fused_conv;
    } else if (model_type == ModelType::ConvBiasAdd || model_type == ModelType::GroupConvBiasAdd) {
        auto fused_conv =
            std::make_shared<TFusedConvolution>(input_node,
                                                Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                                Constant::create(ov::element::f32, conv_params.bias_shape, {1}),
                                                Constant::create(ov::element::f32, conv_params.eltwise_shape, {2}),
                                                conv_params.stride,
                                                conv_params.pads_begin,
                                                conv_params.pads_end,
                                                conv_params.dilation,
                                                conv_params.pad_type,
                                                act_type);
        last_op = fused_conv;
    } else if (model_type == ModelType::ConvBiasConvBiasAdd || model_type == ModelType::GroupConvBiasConvBiasAdd) {
        auto fused_conv =
            std::make_shared<TFusedConvolution>(input_node,
                                                Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                                Constant::create(ov::element::f32, conv_params.bias_shape, {1}),
                                                conv_params.stride,
                                                conv_params.pads_begin,
                                                conv_params.pads_end,
                                                conv_params.dilation,
                                                conv_params.pad_type,
                                                act_type);
        auto fused_conv2 =
            std::make_shared<TFusedConvolution>(fused_conv,
                                                Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                                Constant::create(ov::element::f32, conv_params.bias_shape, {1}),
                                                fused_conv,
                                                conv_params.stride,
                                                conv_params.pads_begin,
                                                conv_params.pads_end,
                                                conv_params.dilation,
                                                conv_params.pad_type,
                                                act_type);
        last_op = fused_conv2;
    } else if (model_type == ModelType::ConvBiasMulAdd || model_type == ModelType::GroupConvBiasMulAdd) {
        auto fused_conv =
            std::make_shared<TFusedConvolution>(input_node,
                                                Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
                                                Constant::create(ov::element::f32, conv_params.bias_shape, {1}),
                                                conv_params.stride,
                                                conv_params.pads_begin,
                                                conv_params.pads_end,
                                                conv_params.dilation,
                                                conv_params.pad_type,
                                                act_type);
        auto mul =
            std::make_shared<Multiply>(fused_conv, Constant::create(ov::element::f32, conv_params.eltwise_shape, {1}));
        auto add = std::make_shared<Add>(fused_conv, mul);
        last_op = add;
    }
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<Result>(last_op)},
                                       ov::ParameterVector{input_node});
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------

template <typename TFusedConvolution>
class FuseConvTestCommonFixture
    : public ::testing::WithParamInterface<
          std::tuple<ModelType /* model type */, ActivationMode /* act type */, ConvParams /* conv params */>>,
      public TransformationTestsF {
public:
    ModelType model_type;
    ActivationMode act_type;

    void Execute() {
        ConvParams conv_params;
        std::tie(model_type, act_type, conv_params) = this->GetParam();
        { model = createModel(model_type, act_type, conv_params); }

        {
            model_ref = (conv_params.bias_shape == conv_params.eltwise_shape)
                            ? model->clone()
                            : createRefModel<TFusedConvolution>(model_type, act_type, conv_params);
        }
        manager.register_pass<ov::nvidia_gpu::pass::CudaConvolutionFusion>();
        manager.run_passes(model);
    }
    static std::string getTestCaseName(
        testing::TestParamInfo<
            std::tuple<ModelType /* model type */, ActivationMode /* act type */, ConvParams /* conv params */>> obj) {
        const ModelType model_type = std::get<0>(obj.param);
        const ActivationMode act_type = std::get<1>(obj.param);
        const ConvParams conv_params = std::get<2>(obj.param);

        std::ostringstream result;
        if (model_type == ModelType::ConvBias) {
            result << "ConvBias";
        } else if (model_type == ModelType::ConvBiasAdd) {
            result << "ConvBiasAdd";
        } else if (model_type == ModelType::ConvBiasConvBiasAdd) {
            result << "ConvBiasConvBiasAdd";
        } else if (model_type == ModelType::ConvBiasMulAdd) {
            result << "ConvBiasMulAdd";
        } else if (model_type == ModelType::GroupConvBias) {
            result << "GroupConvBias";
        } else if (model_type == ModelType::GroupConvBiasAdd) {
            result << "GroupConvBiasAdd";
        } else if (model_type == ModelType::GroupConvBiasConvBiasAdd) {
            result << "GroupConvBiasConvBiasAdd";
        } else if (model_type == ModelType::GroupConvBiasMulAdd) {
            result << "GroupConvBiasMulAdd";
        }
        result << "_IS" << conv_params.input_shape;
        result << "_FS" << conv_params.filter_shape;
        result << "_BS" << conv_params.bias_shape;
        result << "_ES" << conv_params.eltwise_shape;
        result << "_" << conv_params.stride;
        result << "_" << conv_params.dilation;
        result << "_" << conv_params.pads_begin;
        result << "_" << conv_params.pads_end;
        result << "_" << conv_params.pad_type;
        if (act_type == ActivationMode::RELU) {
            result << "_RELU";
        } else if (act_type == ActivationMode::SIGMOID) {
            result << "_SIGMOID";
        }
        return result.str();
    }
};

using FuseConvTestFixture = FuseConvTestCommonFixture<FusedConvolution>;

TEST_P(FuseConvTestFixture, CompareFunctions) { Execute(); }

INSTANTIATE_TEST_SUITE_P(FuseConvTestSuite,
                         FuseConvTestFixture,
                         ::testing::Combine(::testing::ValuesIn(conv_model_types),
                                            ::testing::ValuesIn(conv_activation_types),
                                            ::testing::ValuesIn(std::vector<ConvParams>{{ov::Shape{1, 3, 64, 64},
                                                                                         ov::Shape{3, 3, 1, 1},
                                                                                         ov::Shape{1, 3, 1, 1},
                                                                                         ov::Shape{1, 3, 64, 64},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::op::PadType::AUTO},
                                                                                        {ov::Shape{1, 3, 64, 64},
                                                                                         ov::Shape{3, 3, 1, 1},
                                                                                         ov::Shape{1, 3, 64, 64},
                                                                                         ov::Shape{1, 3, 64, 64},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::op::PadType::AUTO},
                                                                                        {ov::Shape{8, 3, 64, 64},
                                                                                         ov::Shape{3, 3, 1, 1},
                                                                                         ov::Shape{1, 3, 1, 1},
                                                                                         ov::Shape{8, 3, 64, 64},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::op::PadType::AUTO}})),
                         FuseConvTestFixture::getTestCaseName);

using FuseGroupConvTestFixture = FuseConvTestCommonFixture<FusedGroupConvolution>;

TEST_P(FuseGroupConvTestFixture, CompareFunctions) { Execute(); }

INSTANTIATE_TEST_SUITE_P(FuseConvTestSuite,
                         FuseGroupConvTestFixture,
                         ::testing::Combine(::testing::ValuesIn(group_conv_model_types),
                                            ::testing::ValuesIn(group_conv_activation_types),
                                            ::testing::ValuesIn(std::vector<ConvParams>{{ov::Shape{1, 32, 112, 112},
                                                                                         ov::Shape{32, 1, 1, 3, 3},
                                                                                         ov::Shape{1, 32, 1, 1},
                                                                                         ov::Shape{1, 32, 112, 112},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::op::PadType::AUTO},
                                                                                        {ov::Shape{1, 32, 112, 112},
                                                                                         ov::Shape{32, 1, 1, 3, 3},
                                                                                         ov::Shape{1, 32, 112, 112},
                                                                                         ov::Shape{1, 32, 112, 112},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::op::PadType::AUTO},
                                                                                        {ov::Shape{8, 32, 112, 112},
                                                                                         ov::Shape{32, 1, 1, 3, 3},
                                                                                         ov::Shape{1, 32, 1, 1},
                                                                                         ov::Shape{8, 32, 112, 112},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::Strides{1, 1},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::CoordinateDiff{0, 0},
                                                                                         ov::op::PadType::AUTO}})),
                         FuseGroupConvTestFixture::getTestCaseName);

// ---------------------------------------------------------------------------------------------------------------------

namespace {
struct ConvBackPropParams {
    ov::Shape input_shape;
    ov::Shape filter_shape;
    ov::Shape eltwise_shape;
    ov::Strides stride;
    ov::Strides dilation;
    ov::CoordinateDiff pads_begin;
    ov::CoordinateDiff pads_end;
    ov::op::PadType pad_type;
    ov::CoordinateDiff output_pad;
};

std::shared_ptr<Result> createModelBackprop(const ModelType& model_type,
                                            const ov::Output<ov::Node>& input_node,
                                            const ConvBackPropParams& conv_params) {
    std::shared_ptr<ov::Node> last_op = nullptr;

    auto add_conv_backprop = [&](const ov::Output<ov::Node>& output) -> std::shared_ptr<ConvolutionBackpropData> {
        return std::make_shared<ConvolutionBackpropData>(
            output,
            Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
            conv_params.stride,
            conv_params.pads_begin,
            conv_params.pads_end,
            conv_params.dilation,
            conv_params.pad_type,
            conv_params.output_pad);
    };

    auto add_eltwise = [&](const ov::Output<ov::Node>& output) -> std::shared_ptr<Add> {
        return std::make_shared<Add>(output, Constant::create(ov::element::f32, conv_params.eltwise_shape, {1}));
    };

    if (model_type == ModelType::ConvBackpropAdd) {
        auto conv = add_conv_backprop(input_node);
        auto bias = add_eltwise(conv);
        last_op = bias;
    }
    return std::make_shared<Result>(last_op);
}

std::shared_ptr<Result> createRefModelBackprop(const ModelType& model_type,
                                               const ov::Output<ov::Node>& input_node,
                                               const ConvBackPropParams& conv_params) {
    std::shared_ptr<ov::Node> last_op = nullptr;
    if (model_type == ModelType::ConvBackpropAdd) {
        auto fused_conv = std::make_shared<FusedConvBackpropData>(
            input_node,
            Constant::create(ov::element::f32, conv_params.filter_shape, {0.01}),
            Constant::create(ov::element::f32, conv_params.eltwise_shape, {1}),
            conv_params.stride,
            conv_params.pads_begin,
            conv_params.pads_end,
            conv_params.dilation,
            conv_params.pad_type,
            conv_params.output_pad);
        last_op = fused_conv;
    }
    return std::make_shared<Result>(last_op);
}
}  // namespace

class FuseConvBackPropTestFixture : public ::testing::WithParamInterface<
                                        std::tuple<ModelType /* model type */, ConvBackPropParams /* conv params */>>,
                                    public TransformationTestsF {
public:
    ModelType model_type;

    void Execute() {
        ConvBackPropParams conv_params;
        std::tie(model_type, conv_params) = this->GetParam();
        {
            auto input_params = std::make_shared<Parameter>(ov::element::f32, conv_params.input_shape);
            auto result = createModelBackprop(model_type, input_params, conv_params);
            model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params});
        }

        {
            auto input_params = std::make_shared<Parameter>(ov::element::f32, conv_params.input_shape);
            auto result = createRefModelBackprop(model_type, input_params, conv_params);
            model_ref = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input_params});
        }
        manager.register_pass<ov::nvidia_gpu::pass::CudaConvolutionFusion>();
        manager.run_passes(model);
    }
    static std::string getTestCaseName(
        testing::TestParamInfo<std::tuple<ModelType /* model type */, ConvBackPropParams /* conv params */>> obj) {
        const ModelType model_type = std::get<0>(obj.param);
        const ConvBackPropParams conv_params = std::get<1>(obj.param);
        std::ostringstream result;
        if (model_type == ModelType::ConvBackpropAdd) {
            result << "ConvBackpropAdd";
        }
        result << "_" << conv_params.stride;
        result << "_" << conv_params.dilation;
        result << "_" << conv_params.pads_begin;
        result << "_" << conv_params.pads_end;
        result << "_" << conv_params.pad_type;
        result << "_" << conv_params.output_pad;
        return result.str();
    }
};

TEST_P(FuseConvBackPropTestFixture, CompareFunctions) { Execute(); }

INSTANTIATE_TEST_SUITE_P(FuseConvTestSuite,
                         FuseConvBackPropTestFixture,
                         ::testing::Combine(::testing::ValuesIn(conv_backprop_model_types),
                                            ::testing::Values(ConvBackPropParams{ov::Shape{1, 512, 32, 32},
                                                                                 ov::Shape{512, 256, 4, 4},
                                                                                 ov::Shape{1, 256, 64, 64},
                                                                                 ov::Strides{2, 2},
                                                                                 ov::Strides{1, 1},
                                                                                 ov::CoordinateDiff{1, 1},
                                                                                 ov::CoordinateDiff{1, 1},
                                                                                 ov::op::PadType::EXPLICIT,
                                                                                 ov::CoordinateDiff{0, 0}})),
                         FuseConvBackPropTestFixture::getTestCaseName);
}  // namespace testing
