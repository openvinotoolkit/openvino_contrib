// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include "quantize_fusion.hpp"

#include <memory>
#include <numeric>
#include <vector>
#include <cstdint>

#include <ie_algorithm.hpp>

#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "opset/opset.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pass/manager.hpp>

using namespace ArmPlugin;
using Types = std::vector<ngraph::element::Type>;
namespace {
std::vector<float> getFloatVector(const opset::Constant& constant) {
    auto outputType = constant.get_output_element_type(0);
    if (outputType == ngraph::element::f32) {
        return constant.cast_vector<float>();
    } else if (outputType == ngraph::element::f16) {
        auto vec = constant.cast_vector<ngraph::float16>();
        return {std::begin(vec), std::end(vec)};
    } else {
        IE_THROW() << "Unsupported element type: " << outputType;
    }
}

std::vector<std::int32_t> getIntVector(const opset::Constant& constant) {
    auto outputType = constant.get_output_element_type(0);
    std::vector<std::int32_t> result;
    if (outputType == ngraph::element::f32) {
        for (auto& v : constant.cast_vector<float>()) {
            result.emplace_back(std::round(v));
        }
    } else if (outputType == ngraph::element::f16) {
        for (auto& v : constant.cast_vector<ngraph::float16>()) {
            result.emplace_back(std::round(static_cast<float>(v)));
        }
    } else if (outputType == ngraph::element::i8) {
        for (auto& v : constant.cast_vector<std::int8_t>()) {
            result.emplace_back(v);
        }
    } else if (outputType == ngraph::element::u8) {
        for (auto& v : constant.cast_vector<std::uint8_t>()) {
            result.emplace_back(v);
        }
    }  else {
        IE_THROW() << "Unsupported element type: " << outputType;
    }
    return result;
}

std::shared_ptr<ngraph::Node> makeTypeRelaxed(const ngraph::Node* node,
                                              const std::vector<ngraph::Output<ngraph::Node>>& newInputs,
                                              const Types& inputTypes,
                                              const Types& outputTypes) {
#define CASE(TYPE)                                                                  \
    if (ngraph::is_type<TYPE>(node)) {                                              \
        return std::make_shared<ngraph::op::TypeRelaxed<TYPE>>(                     \
            *std::static_pointer_cast<TYPE>(node->copy_with_new_inputs(newInputs)), \
            inputTypes,                                                             \
            outputTypes);                                                           \
    }

    CASE(opset::ArmConvolution)
    CASE(opset::ArmGroupConvolution)
    CASE(opset::MatMul)
    CASE(opset::AvgPool)
    CASE(opset::ReduceMean)
    CASE(opset::ArmNoOp)
    IE_ASSERT(!"Arm Plugin: Unregistered type: ") << node;
#undef CASE
}
}  // namespace

ArmPlugin::pass::ConvertQuantize::ConvertQuantize() {
    auto fakeQuantize = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(
        std::make_shared<ngraph::pattern::Matcher>(fakeQuantize, "ConvertQuantize"),
        [](ngraph::pattern::Matcher& m) {
            auto fakeQuantize = std::dynamic_pointer_cast<opset::FakeQuantize>(m.get_match_root());
            IE_ASSERT(fakeQuantize != nullptr);
            auto input_type = fakeQuantize->input(0).get_element_type();
            auto output_type = fakeQuantize->output(0).get_element_type();
            auto input = fakeQuantize->input_value(0);
            auto input_low = fakeQuantize->input_value(1);
            auto input_high = fakeQuantize->input_value(2);
            auto output_low = fakeQuantize->input_value(3);
            auto output_high = fakeQuantize->input_value(4);
            using Types = std::vector<ngraph::element::Type>;
            if ((input_type.is_real() || input_type.is_quantized()) && output_type.is_quantized()) {
                auto quantizationInfo = opset::makeQuantizationInfo(input_low, input_high, output_low, output_high);
                auto scales = quantizationInfo.scale();
                auto offsets = quantizationInfo.offset();
                if (scales.size() > 1) {
                    auto axis  = opset::Constant::create<int64_t>(ngraph::element::i64, ngraph::Shape{}, {1});
                    auto split = std::make_shared<opset::Split>(fakeQuantize->input_value(0), axis, scales.size());
                    ngraph::copy_runtime_info(fakeQuantize, split);
                    split->set_friendly_name(fakeQuantize->get_friendly_name() + "_split");
                    ngraph::NodeVector concat_inputs;
                    for (std::size_t i = 0; i < scales.size(); ++i) {
                        auto armQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmQuantize>>(
                            Types{input_type}, Types{output_type}, split->output(i));
                        armQuantize->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize_" + std::to_string(i));
                        ngraph::copy_runtime_info(fakeQuantize, armQuantize);
                        armQuantize->get_rt_info().emplace("QuantizationInfo",
                            std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                arm_compute::QuantizationInfo{scales[i], offsets[i]}));
                        concat_inputs.push_back(armQuantize);
                    }

                    auto concat = std::make_shared<opset::Concat>(concat_inputs, 1);
                    concat->set_friendly_name(fakeQuantize->get_friendly_name() + "_concat");
                    ngraph::copy_runtime_info(fakeQuantize, concat);
                    concat->get_rt_info().emplace("QuantizationInfo",
                        std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                            arm_compute::QuantizationInfo{1, 0}));
                    ngraph::replace_node(fakeQuantize, concat);
                } else {
                    auto armQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmQuantize>>(Types{input_type}, Types{output_type}, input);
                    armQuantize->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize");
                    ngraph::copy_runtime_info(fakeQuantize, armQuantize);
                    armQuantize->get_rt_info()["QuantizationInfo"] =
                        std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(quantizationInfo);
                    auto noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                        Types{output_type}, Types{output_type},
                        armQuantize);
                    noOp->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_noop");
                    noOp->get_rt_info().emplace("QuantizationInfo",
                        std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                            arm_compute::QuantizationInfo{1, 0}));
                    ngraph::replace_node(fakeQuantize, noOp);
                }
                return true;
            }
            return false;
        });
}

ArmPlugin::pass::NodeQuantizeFusion::NodeQuantizeFusion() {
    auto node_pattern = ngraph::pattern::wrap_type<
        opset::ArmConvolution,
        opset::ArmGroupConvolution,
        opset::MatMul,
        opset::AvgPool,
        opset::ReduceMean>(ngraph::pattern::consumers_count(1));
    auto activation_pattern = ngraph::pattern::wrap_type<
        opset::Sigmoid, opset::Tanh, opset::Relu, opset::Abs,
        opset::Elu, opset::Sqrt, opset::SoftPlus, opset::HSwish,
        opset::PRelu, opset::Clamp>({node_pattern});
    auto node_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{node_pattern, activation_pattern});
    auto fq_pattern = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        node_output,
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fq_pattern, "NodeQuantizeFusion"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto node = pattern_map[node_pattern].get_node_shared_ptr();
            auto fakeQuantize = ngraph::as_type_ptr<opset::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
            auto itActivation = pattern_map.find(activation_pattern);
            auto realType = node->get_output_element_type(0);
            auto quantizedType = fakeQuantize->get_output_element_type(0);
            if (!(realType.is_real() && quantizedType.is_quantized())) {
                return false;
            }
            std::vector<ngraph::Output<ngraph::Node>> newInputs;
            Types inputTypes;
            for (auto&& input : node->inputs()) {
                inputTypes.emplace_back(realType);
                newInputs.emplace_back(
                    ngraph::op::TemporaryReplaceOutputType{input.get_source_output(), realType}.get());
            }
            auto newNode = makeTypeRelaxed(node.get(), newInputs, inputTypes, Types{quantizedType});
            auto noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                Types{quantizedType}, Types{quantizedType},
                newNode);
            noOp->set_friendly_name(node->get_friendly_name() + "_arm_noop_output");
            noOp->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                    arm_compute::QuantizationInfo{1, 0}));
            if (itActivation != pattern_map.end()) {
                auto activation = itActivation->second.get_node_shared_ptr();
                auto activationLayerInfo = opset::makeActivationLayerInfo(activation.get());
                newNode->set_friendly_name(node->get_friendly_name() + '_' +
                                           activation->get_friendly_name() + '_' +
                                           fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info({node, activation, fakeQuantize}, newNode);
                if (activationLayerInfo.enabled()) {
                    newNode->get_rt_info().emplace("ActivationLayerInfo",
                        std::make_shared<ngraph::VariantWrapper<arm_compute::ActivationLayerInfo>>(activationLayerInfo));
                }
            } else {
                newNode->set_friendly_name(node->get_friendly_name() + '_' + fakeQuantize->get_friendly_name());
            }
            auto quantizationInfo = opset::makeQuantizationInfo(fakeQuantize->input_value(1), fakeQuantize->input_value(2),
                                                                fakeQuantize->input_value(3), fakeQuantize->input_value(4));

            newNode->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                    quantizationInfo));
            ngraph::replace_node(fakeQuantize, noOp);
            return true;
        });
}

struct DequantizeNodeFusionBase : ngraph::pass::MatcherPass {
DequantizeNodeFusionBase(bool mulOnly, const std::string& name) {
    auto scale_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset::Multiply>(
        {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()), scale_pattern},
        ngraph::pattern::consumers_count(1));

    auto mul_add_offset_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_add_pattern = ngraph::pattern::wrap_type<opset::Add>(
        {mul_pattern, mul_add_offset_pattern},
        ngraph::pattern::consumers_count(1));

    auto mul_sub_offset_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_sub_pattern = ngraph::pattern::wrap_type<opset::Subtract>(
        {mul_pattern, mul_sub_offset_pattern},
        ngraph::pattern::consumers_count(1));

    auto sub_offset_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto sub_pattern = ngraph::pattern::wrap_type<opset::Subtract>(
        {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()), sub_offset_pattern},
        ngraph::pattern::consumers_count(1));

    auto dequantize_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{
       sub_pattern, mul_add_pattern, mul_sub_pattern});
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(mulOnly ? mul_pattern : dequantize_output, name),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto itMulAdd = pattern_map.find(mul_add_pattern);
            auto itMulSub = pattern_map.find(mul_sub_pattern);
            auto itMul = pattern_map.find(mul_pattern);
            auto itSub = pattern_map.find(sub_pattern);
            auto output = [&] {
                for (auto& it : {itMulAdd, itMulSub, itMul, itSub}) if (it != pattern_map.end()) return it->second.get_node_shared_ptr();
                IE_ASSERT(!"Arm Plugin: No output pattern found!");
            }();
            if (output->output(0).get_target_inputs().size() != 1) {
                return false;
            }
            auto input = [&] {
                for (auto& it : {itMul, itSub}) if (it != pattern_map.end()) return it->second.get_node_shared_ptr();
                IE_ASSERT(!"Arm Plugin: No input pattern found!");
            }();

            auto quantizedType = input->get_input_element_type(0);
            auto realType = output->get_output_element_type(0);

            if (!(quantizedType.is_quantized() && realType.is_real())) {
                return false;
            }

            if (input->get_input_element_type(0) != ngraph::element::i8 && input->get_input_element_type(0) != ngraph::element::u8) {
                auto convert = std::make_shared<opset::Convert>(input->input_value(0), realType);
                convert->set_friendly_name(input->get_friendly_name() + "_input_convert");
                ngraph::insert_new_node_between(
                    input->input_value(0).get_node_shared_ptr(),
                    input,
                    convert);
            } else {
                auto noOp = input->input_value(0).get_node_shared_ptr();
                if (!ngraph::is_type<opset::ArmNoOp>(noOp) ||
                    (ngraph::is_type<opset::ArmNoOp>(noOp) && (noOp->output(0).get_target_inputs().size() != 1))) {
                    noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                        Types{quantizedType}, Types{quantizedType},
                        input->input_value(0));
                    noOp->set_friendly_name(input->get_friendly_name() + "_arm_noop");
                }

                std::vector<std::int32_t> offsets;
                std::vector<float> scales;
                if ((itMulSub != pattern_map.end()) || (itMulAdd != pattern_map.end())) {
                    scales = getFloatVector(*ngraph::as_type<opset::Constant>(pattern_map[scale_pattern].get_node()));
                    auto floatOffsets = getFloatVector(*ngraph::as_type<opset::Constant>(pattern_map[
                        (itMulSub != pattern_map.end()) ? mul_sub_offset_pattern : mul_add_offset_pattern
                    ].get_node()));
                    offsets.resize(scales.size());
                    for (std::size_t i = 0; i < scales.size(); ++i) {
                        offsets[i] = -std::round(floatOffsets[i]/scales[i]);
                    }
                } else if (itSub != pattern_map.end()) {
                    offsets = getIntVector(*ngraph::as_type<opset::Constant>(pattern_map[sub_offset_pattern].get_node()));
                    scales.resize(offsets.size(), 1.0);
                } else if (itMul != pattern_map.end()) {
                    scales = getFloatVector(*ngraph::as_type<opset::Constant>(pattern_map[scale_pattern].get_node()));
                    offsets.resize(scales.size(), 0);
                }

                auto node = output->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                std::shared_ptr<ngraph::Node> newNode;
                std::shared_ptr<ngraph::Node> nodeToReplace;
                if ((ngraph::is_type<opset::ArmConvolution>(node) ||
                    ngraph::is_type<opset::ArmGroupConvolution>(node) ||
                    ngraph::is_type<opset::MatMul>(node) ||
                    ngraph::is_type<opset::AvgPool>(node) ||
                    ngraph::is_type<opset::ReduceMean>(node)) && node->get_output_element_type(0).is_quantized())  {
                    std::vector<ngraph::Output<ngraph::Node>> newInputs;
                    Types inputTypes;
                    for (auto&& input : node->inputs()) {
                        inputTypes.emplace_back(realType);
                        newInputs.emplace_back(
                            ngraph::op::TemporaryReplaceOutputType{input.get_source_output(), realType}.get());
                    }
                    newInputs[0] = ngraph::op::TemporaryReplaceOutputType{noOp->output(0), realType}.get();
                    newNode = makeTypeRelaxed(node.get(), newInputs, inputTypes, Types{node->get_output_element_type(0)});
                    newNode->set_friendly_name(node->get_friendly_name());
                    nodeToReplace = node;
                } else if ((scales.size() == 1) && (offsets.size() == 1)) {
                    newNode = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(Types{quantizedType}, Types{realType}, noOp);
                    newNode->set_friendly_name(output->get_friendly_name() + "_arm_dequantize");
                    nodeToReplace = output;
                } else {
                    return false;
                }

                std::vector<std::shared_ptr<ngraph::Node>> nodesToReplace;
                for (auto& pattern : {scale_pattern, mul_pattern,
                                      mul_add_offset_pattern, mul_add_pattern,
                                      mul_sub_offset_pattern, mul_sub_pattern,
                                      sub_offset_pattern, sub_pattern}) {
                    auto itPattern = pattern_map.find(pattern);
                    if (itPattern != pattern_map.end()) {
                        nodesToReplace.emplace_back(itPattern->second.get_node_shared_ptr());
                    }
                }
                ngraph::copy_runtime_info(nodesToReplace, noOp);
                noOp->get_rt_info()["QuantizationInfo"] =
                    std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                        arm_compute::QuantizationInfo{scales, offsets});


                ngraph::copy_runtime_info(nodeToReplace, newNode);
                ngraph::replace_node(nodeToReplace, newNode);
            }

            return true;
        });
}
};

struct DequantizeNodeFusionPattern : public DequantizeNodeFusionBase{
    DequantizeNodeFusionPattern() : DequantizeNodeFusionBase{false, "DequantizeNodeFusionPattern"} {}
};

struct DequantizeNodeFusionMul : public DequantizeNodeFusionBase{
    DequantizeNodeFusionMul() : DequantizeNodeFusionBase{true, "DequantizeNodeFusionMul"} {}
};

bool ArmPlugin::pass::DequantizeNodeFusion::run_on_function(std::shared_ptr<ngraph::Function> f) {
    ngraph::pass::Manager manager;
    manager.register_pass<DequantizeNodeFusionPattern>();
    manager.register_pass<DequantizeNodeFusionMul>();
    manager.run_passes(f);
    return true;
}

bool ArmPlugin::pass::PropogateQuantizationInfo::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto&& node : f->get_ordered_ops()) {
        auto outputs = node->outputs();
        auto quantizedOutput = std::any_of(std::begin(outputs), std::end(outputs), [] (auto& output) {
            return output.get_element_type().is_quantized();
        });
        if (quantizedOutput) {
            node->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                arm_compute::QuantizationInfo{1, 0}));
        }
    }
    return false;
}

ArmPlugin::pass::AddDequantizeOnInputs::AddDequantizeOnInputs() {
    auto node_pattern = ngraph::pattern::wrap_type<
        opset::ArmConvolution,
        opset::ArmGroupConvolution,
        opset::MatMul,
        opset::Add,
        opset::Subtract,
        opset::Multiply>();
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(node_pattern, "AddDequantizeOnInputs"),
        [](ngraph::pattern::Matcher& m) {
            auto node = m.get_match_root();

            auto inputs = node->inputs();
            if (node->outputs().size() != 1) {
                return false;
            }
            auto outputType = node->get_output_element_type(0);

            auto nodeHasQuantizedInputs = std::any_of(std::begin(inputs), std::end(inputs), [] (auto& input) {
                return input.get_element_type().is_quantized();
            });

            bool result = false;
            if (nodeHasQuantizedInputs && outputType.is_real()) {
                for (auto&& input : inputs) {
                    auto inputType = input.get_element_type();
                    if (inputType.is_quantized()) {
                        std::shared_ptr<ngraph::Node> newInputOp;
                        if (ngraph::op::is_constant(input.get_source_output().get_node())) {
                            newInputOp = std::make_shared<opset::Convert>(input.get_source_output(), outputType);
                        } else {
                            auto constant = std::make_shared<opset::Constant>(outputType, ngraph::Shape{1});
                            auto noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                                Types{inputType}, Types{inputType},
                                input.get_source_output());
                            noOp->set_friendly_name(node->get_friendly_name() + "_on_input_" + std::to_string(input.get_index()) + "_arm_noop");
                            noOp->get_rt_info()["QuantizationInfo"] =
                                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                    arm_compute::QuantizationInfo{1, 0});
                            newInputOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(Types{inputType}, Types{outputType}, noOp);
                            newInputOp->set_friendly_name(node->get_friendly_name() + "_on_input_" + std::to_string(input.get_index()) + "_arm_dequantize");
                        }
                        ngraph::insert_new_node_between(
                            input.get_source_output().get_node_shared_ptr(),
                            node,
                            newInputOp);
                        result = true;
                    }
                }
            }
            return result;
        });
}

ArmPlugin::pass::ConvertBiasToI32::ConvertBiasToI32() {
    auto conv = ngraph::pattern::wrap_type<
        opset::ArmConvolution,
        opset::ArmGroupConvolution,
        opset::MatMul>();
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(conv, "ConvertBiasToI32"), [](ngraph::pattern::Matcher& m) {
        auto conv = m.get_match_root();
        if (conv->inputs().size() != 3) {
            return false;
        }
        if (!(conv->get_input_element_type(0).is_quantized() ||
              conv->get_input_element_type(1).is_quantized())) {
            return false;
        }
        ngraph::insert_new_node_between(
            conv->input_value(2).get_node_shared_ptr(),
            conv,
            std::make_shared<opset::Convert>(conv->input_value(2), ngraph::element::i32));
        return true;
    });
}

ArmPlugin::pass::MovePerChenelQuantizationInfoToWeights::MovePerChenelQuantizationInfoToWeights() {
    auto node_pattern = ngraph::pattern::wrap_type<
        opset::ArmConvolution,
        opset::ArmGroupConvolution,
        opset::MatMul>();
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(node_pattern, "MovePerChenelQuantizationInfoToWeights"),
    [](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto weights = node->input_value(1).get_node();
        auto itInfo = node->get_rt_info().find("QuantizationInfo");
        if (itInfo != node->get_rt_info().end()) {
            auto quantizationInfo = std::dynamic_pointer_cast<ngraph::VariantImpl<arm_compute::QuantizationInfo>>(itInfo->second)->get();
            auto scales = quantizationInfo.scale();
            auto offsets = quantizationInfo.offset();
            if (scales.size() > 1) {
                std::vector<float> invScales;
                for (auto& scale : scales) {
                    invScales.push_back(1.f/scale);
                }

                weights->get_rt_info()["QuantizationInfo"] =
                            std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                arm_compute::QuantizationInfo{invScales, std::vector<std::int32_t>(offsets.size(), 0)});

                auto allOffsetsAreTheSame = std::all_of(std::begin(offsets), std::end(offsets), [&] (auto offset) {
                    return offset == offsets.front();
                });

                if (allOffsetsAreTheSame) {
                    node->get_rt_info()["QuantizationInfo"] =
                                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                    arm_compute::QuantizationInfo{1, offsets.front()});
                } else {
                    std::vector<ngraph::Output<ngraph::Node>> newInputs;
                    Types inputTypes;
                    for (auto&& input : node->inputs()) {
                        inputTypes.emplace_back(ngraph::element::f32);
                        newInputs.emplace_back(
                            ngraph::op::TemporaryReplaceOutputType{input.get_source_output(), ngraph::element::f32}.get());
                    }
                    auto shape = ngraph::Shape{{offsets.size()}};
                    std::shared_ptr<ngraph::Node> bias;
                    bias = std::make_shared<opset::Convert>(
                        std::make_shared<opset::Multiply>(
                            std::make_shared<opset::Convert>(std::make_shared<opset::Constant>(ngraph::element::i32, shape, offsets),
                                                            ngraph::element::f32),
                            std::make_shared<opset::Constant>(ngraph::element::f32, shape, scales)),
                        ngraph::element::i32);
                    if (node->inputs().size() > 2) {
                        bias = std::make_shared<opset::Add>(node->input_value(2), bias);
                        newInputs[2] = ngraph::op::TemporaryReplaceOutputType{bias->output(0), ngraph::element::f32}.get();
                    } else {
                        inputTypes.emplace_back(ngraph::element::f32);
                        newInputs.emplace_back(ngraph::op::TemporaryReplaceOutputType{bias->output(0), ngraph::element::f32}.get());
                    }

                    auto newNode = makeTypeRelaxed(node.get(), newInputs, inputTypes, Types{node->get_output_element_type(0)});
                    newNode->set_friendly_name(node->get_friendly_name());
                    ngraph::copy_runtime_info(node, newNode);
                    newNode->get_rt_info()["QuantizationInfo"] =
                                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                    arm_compute::QuantizationInfo{1, 0});
                    ngraph::replace_node(node, newNode);
                }
                return true;
            }
        }
        return false;
    });
}


