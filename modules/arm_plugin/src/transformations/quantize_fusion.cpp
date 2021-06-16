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

using namespace ArmPlugin;
using Types = std::vector<ngraph::element::Type>;

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
                auto armQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmQuantize>>(
                    Types{input_type}, Types{output_type},
                    input, input_low, input_high, output_low, output_high, fakeQuantize->get_levels(), fakeQuantize->get_auto_broadcast());
                armQuantize->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize");
                ngraph::copy_runtime_info(fakeQuantize, armQuantize);
                armQuantize->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(quantizationInfo));
                auto noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                    Types{output_type}, Types{output_type},
                    armQuantize);
                noOp->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_noop");
                noOp->get_rt_info().emplace("QuantizationInfo",
                    std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                        arm_compute::QuantizationInfo{1, 0}));
                ngraph::replace_node(fakeQuantize, noOp);
                return true;
            }
            return false;
        });
}


template <class Node>
void ArmPlugin::pass::FakeQuantizeFusionBase::registerMatcher(const std::string& name, bool withActivation) {
    auto node_pattern = ngraph::pattern::wrap_type<Node>(ngraph::pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        (!withActivation) ? node_pattern : ngraph::pattern::wrap_type<
            opset::Sigmoid, opset::Tanh, opset::Relu, opset::Abs,
            opset::Elu, opset::Sqrt, opset::SoftPlus, opset::HSwish,
            opset::PRelu, opset::Clamp>({node_pattern}),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fq_pattern, name),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto node = pattern_map[node_pattern].get_node_shared_ptr();
            auto fakeQuantize = ngraph::as_type_ptr<opset::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
            if (node->output(0).get_target_inputs().size() != 1) {
                return false;
            }
            auto fqInputType = fakeQuantize->get_input_element_type(1);
            IE_ASSERT(fqInputType.is_real());
            Types inputTypes;
            std::vector<ngraph::Output<ngraph::Node>> newInputs;
            for (auto&& input : node->inputs()) {
                inputTypes.emplace_back(fqInputType);
                newInputs.emplace_back(
                    ngraph::op::TemporaryReplaceOutputType{input.get_source_output(), fqInputType}.get());
            }
            auto fqOutputType = fakeQuantize->get_output_element_type(0);
            IE_ASSERT(!fqOutputType.is_real());
            auto quantizedType = fqOutputType;
            auto itMaybeQuantized = node->get_rt_info().find("MayBeQuanitzed");
            if (node->get_input_element_type(0).is_quantized()) {
                quantizedType = node->get_input_element_type(0);
            } else if (node->inputs().size() > 1) {
                if (node->get_input_element_type(1).is_quantized()) {
                    quantizedType = node->get_input_element_type(1);
                }
            } else if (itMaybeQuantized != node->get_rt_info().end()) {
                auto type = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::element::Type>>(itMaybeQuantized->second);
                IE_ASSERT(type != nullptr);
                quantizedType = type->get();
            }
            if (auto substruct = ngraph::as_type<opset::Subtract>(node->input_value(0).get_node())) {
                if (substruct->get_input_element_type(0).is_quantized() && substruct->get_input_element_type(1).is_quantized() &&
                    substruct->get_output_element_type(0).is_real()) {
                    auto constant = ngraph::as_type<opset::Constant>(substruct->input_value(1).get_node());
                    IE_ASSERT(constant->get_output_element_type(0) == ngraph::element::i8);
                    auto offset = static_cast<const std::int8_t*>(constant->get_data_ptr());
                    auto scales = std::vector<float>(ngraph::shape_size(constant->get_output_shape(0)), 1);
                    auto i32offset = std::vector<std::int32_t>(offset, offset + scales.size());
                    auto noOpOnInput = substruct->input_value(0).get_node_shared_ptr();
                    if (!ngraph::is_type<opset::ArmNoOp>(noOpOnInput) ||
                        (ngraph::is_type<opset::ArmNoOp>(noOpOnInput) && (noOpOnInput->outputs().size() != 1))) {
                        noOpOnInput = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                            Types{quantizedType}, Types{quantizedType},
                            substruct->input_value(0));
                        noOpOnInput->set_friendly_name(node->get_friendly_name() + "_arm_noop_input");
                    }
                    noOpOnInput->get_rt_info()["QuantizationInfo"] =
                        std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                            arm_compute::QuantizationInfo{scales, i32offset});
                    newInputs[0] = ngraph::op::TemporaryReplaceOutputType{noOpOnInput, fqInputType}.get();
                }
            }
            auto new_node = std::make_shared<ngraph::op::TypeRelaxed<Node>>(
                        *std::static_pointer_cast<Node>(node->copy_with_new_inputs(newInputs)),
                        inputTypes,
                        Types{quantizedType});

            auto noOpOnOutput = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                Types{quantizedType}, Types{quantizedType},
                new_node);
            noOpOnOutput->set_friendly_name(new_node->get_friendly_name() + "_arm_noop_output");
            noOpOnOutput->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                    arm_compute::QuantizationInfo{1, 0}));

            if (withActivation) {
                auto activationLayerInfo = opset::makeActivationLayerInfo(fakeQuantize->input_value(0).get_node());
                new_node->set_friendly_name(node->get_friendly_name() + '_' +
                                            fakeQuantize->input_value(0).get_node()->get_friendly_name() + '_' +
                                            fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info({node, fakeQuantize, fakeQuantize->input_value(0).get_node_shared_ptr()}, new_node);
                if (activationLayerInfo.enabled()) {
                    new_node->get_rt_info().emplace("ActivationLayerInfo",
                        std::make_shared<ngraph::VariantWrapper<arm_compute::ActivationLayerInfo>>(activationLayerInfo));
                }
            } else {
                new_node->set_friendly_name(node->get_friendly_name() + '_' + fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info({node, fakeQuantize}, new_node);
            }
            auto quantizationInfo = opset::makeQuantizationInfo(fakeQuantize->input_value(1), fakeQuantize->input_value(2),
                                                                fakeQuantize->input_value(3), fakeQuantize->input_value(4));
            new_node->get_rt_info().emplace("QuantizationInfo",
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(quantizationInfo));

            ngraph::replace_node(fakeQuantize, noOpOnOutput);

            return true;
        });
}

ArmPlugin::pass::ConvFakeQuantizeFusion::ConvFakeQuantizeFusion() {
    registerMatcher<opset::ArmConvolution>("ArmConvFakeQuantizeFusion");
}

ArmPlugin::pass::GroupConvFakeQuantizeFusion::GroupConvFakeQuantizeFusion() {
    registerMatcher<opset::ArmGroupConvolution>("ArmGroupConvFakeQuantizeFusion");
}

ArmPlugin::pass::ConvActivationFakeQuantizeFusion::ConvActivationFakeQuantizeFusion() {
    registerMatcher<opset::ArmConvolution>("ArmConvActivationFakeQuantizeFusion", true);
}

ArmPlugin::pass::GroupConvActivationFakeQuantizeFusion::GroupConvActivationFakeQuantizeFusion() {
    registerMatcher<opset::ArmGroupConvolution>("ArmGroupConvActivationFakeQuantizeFusion", true);
}

ArmPlugin::pass::MatMulFakeQuantizeFusion::MatMulFakeQuantizeFusion() {
    registerMatcher<opset::MatMul>("MatMulFakeQuantizeFusion");
}

ArmPlugin::pass::AvgPoolFakeQuantizeFusion::AvgPoolFakeQuantizeFusion() {
    registerMatcher<opset::AvgPool>("AvgPoolFakeQuantizeFusion");
}

ArmPlugin::pass::ReduceMeanFakeQuantizeFusion::ReduceMeanFakeQuantizeFusion() {
    registerMatcher<opset::ReduceMean>("ReduceMeanFakeQuantizeFusion");
}

bool ArmPlugin::pass::PropogateQuantizationInfo::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto&& node : f->get_ordered_ops()) {
        if (!node->inputs().empty()) {
            if (node->get_input_element_type(0).is_quantized()) {
                auto it_info = node->get_rt_info().find("QuantizationInfo");
                if (it_info == node->get_rt_info().end()) {
                    auto input_it_info = node->get_input_node_ptr(0)->get_rt_info().find("QuantizationInfo");
                    if (input_it_info != node->get_input_node_ptr(0)->get_rt_info().end()) {
                        node->get_rt_info().emplace("QuantizationInfo", input_it_info->second);
                    } else {
                        node->get_rt_info().emplace("QuantizationInfo",
                            std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                            arm_compute::QuantizationInfo{1, 0}));
                    }
                }
            }
        } else if (ngraph::is_type<opset::Constant>(node.get())) {
            if (node->get_output_element_type(0).is_quantized()) {
                auto it_info = node->get_rt_info().find("QuantizationInfo");
                if (it_info == node->get_rt_info().end()) {
                    node->get_rt_info().emplace("QuantizationInfo",
                                                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                                                    arm_compute::QuantizationInfo{1, 0}));
                }
            }
        }
    }
    return false;
}

ArmPlugin::pass::DeqMulAddToArmDequantizeConvert::DeqMulAddToArmDequantizeConvert() {
    auto scale_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset::Multiply>(
        {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()), scale_pattern},
        ngraph::pattern::consumers_count(1));
    auto offset_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto add_pattern = ngraph::pattern::wrap_type<opset::Add>(
        {mul_pattern, offset_pattern},
        ngraph::pattern::consumers_count(1));

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(add_pattern, "DeqMulAddToArmDequantizeConvert"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto scale = ngraph::as_type<opset::Constant>(pattern_map[scale_pattern].get_node());
            auto mul = pattern_map[mul_pattern].get_node_shared_ptr();
            auto offset = ngraph::as_type<opset::Constant>(pattern_map[offset_pattern].get_node());
            auto add = pattern_map[add_pattern].get_node_shared_ptr();

            if (!(mul->get_input_element_type(0).is_quantized() && mul->get_output_element_type(0).is_real())) {
                return false;
            }

            auto inputType = mul->get_input_element_type(0);
            auto outputType = mul->get_output_element_type(0);
            auto noOp = mul->input_value(0).get_node_shared_ptr();

            if (!ngraph::is_type<opset::ArmNoOp>(noOp) ||
                (ngraph::is_type<opset::ArmNoOp>(noOp) && (noOp->outputs().size() != 1))) {
                noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                    Types{inputType}, Types{inputType},
                    mul->input_value(0));
                noOp->set_friendly_name(mul->get_friendly_name() + "_arm_noop");
            }

            auto scaleShape = scale->get_output_shape(0);
            auto fqConst = std::make_shared<opset::Constant>(mul->get_input_element_type(1), scaleShape);

            auto armDequantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(
                Types{inputType},
                Types{outputType},
                noOp,
                fqConst, fqConst, fqConst, fqConst, 256);

            auto scales = getFloatVector(*scale);
            auto floatOffsets = getFloatVector(*offset);
            std::vector<std::int32_t> offsets(scales.size());
            for (std::size_t i = 0; i < scales.size(); ++i) {
                offsets[i] = -std::round(floatOffsets[i]/scales[i]);
            }

            noOp->get_rt_info()["QuantizationInfo"] =
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(arm_compute::QuantizationInfo{scales, offsets});

            ngraph::copy_runtime_info({mul, add}, armDequantize);
            armDequantize->set_friendly_name(add->get_friendly_name());
            ngraph::replace_node(add, armDequantize);
            return true;
        });
}

ArmPlugin::pass::DeqMulToArmDequantizeConvert::DeqMulToArmDequantizeConvert() {
    auto scale_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset::Multiply>(
        {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()), scale_pattern},
        ngraph::pattern::consumers_count(1));

    register_matcher(std::make_shared<ngraph::pattern::Matcher>(mul_pattern, "DeqMulToArmDequantizeConvert"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto scale = ngraph::as_type<opset::Constant>(pattern_map[scale_pattern].get_node());
            auto mul = pattern_map[mul_pattern].get_node_shared_ptr();

            if (!(mul->get_input_element_type(0).is_quantized() && mul->get_output_element_type(0).is_real())) {
                return false;
            }

            auto inputType = mul->get_input_element_type(0);
            auto outputType = mul->get_output_element_type(0);
            auto noOp = mul->input_value(0).get_node_shared_ptr();
            if (!ngraph::is_type<opset::ArmNoOp>(noOp) ||
                (ngraph::is_type<opset::ArmNoOp>(noOp) && (noOp->outputs().size() != 1))) {
                auto noOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmNoOp>>(
                    Types{inputType}, Types{inputType},
                    mul->input_value(0));
                noOp->set_friendly_name(mul->get_friendly_name() + "_arm_noop");
            }
            auto scaleShape = scale->get_output_shape(0);
            auto scaleConst = std::make_shared<opset::Constant>(mul->get_input_element_type(1), scaleShape);

            auto armDequantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(
                        Types{inputType},
                        Types{outputType},
                        noOp,
                        scaleConst, scaleConst, scaleConst, scaleConst, 256);

            auto scales = getFloatVector(*scale);
            noOp->get_rt_info()["QuantizationInfo"] =
                std::make_shared<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(
                    arm_compute::QuantizationInfo{scales});

            ngraph::copy_runtime_info(mul, armDequantize);
            armDequantize->set_friendly_name(mul->get_friendly_name());
            ngraph::replace_node(mul, armDequantize);
            return true;
        });
}

template <class Node>
void ArmPlugin::pass::AddArmDequantizeOnInputsBase::registerMatcher(const std::string& name) {
    auto node_pattern = ngraph::pattern::wrap_type<Node>();
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(node_pattern, name),
        [=](ngraph::pattern::Matcher& m) {
            auto node = m.get_pattern_value_map()[node_pattern].get_node_shared_ptr();

            auto inputs = node->inputs();
            if (node->outputs().size() != 1) {
                return false;
            }
            auto outputType = node->get_output_element_type(0);

            Types inputTypes;
            std::vector<ngraph::Output<ngraph::Node>> newInputs;

            auto nodeHasQuantizedInputs = std::any_of(std::begin(inputs), std::end(inputs), [] (auto& input) {
                return input.get_element_type().is_quantized();
            });

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
                            newInputOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(
                                        Types{inputType},
                                        Types{outputType},
                                        noOp,
                                        constant, constant, constant, constant, 256);
                            newInputOp->set_friendly_name(node->get_friendly_name() + "_on_input_" + std::to_string(input.get_index()) + "_arm_dequantize");
                        }
                        newInputs.emplace_back(
                            ngraph::op::TemporaryReplaceOutputType{newInputOp, outputType}.get());
                    } else {
                        newInputs.emplace_back(input.get_source_output());
                    }
                }
            } else {
                return false;
            }

            auto nodeWithRealInputs = std::make_shared<ngraph::op::TypeRelaxed<Node>>(
                *std::static_pointer_cast<Node>(node->copy_with_new_inputs(newInputs)),
                inputTypes,
                Types{outputType});

            ngraph::copy_runtime_info(node, nodeWithRealInputs);
            nodeWithRealInputs->set_friendly_name(node->get_friendly_name());
            ngraph::replace_node(node, nodeWithRealInputs);

            return true;
        });
}

ArmPlugin::pass::AddArmDequantizeOnInputsConv::AddArmDequantizeOnInputsConv() {
    registerMatcher<opset::ArmConvolution>("AddArmDequantizeOnInputsConv");
}

ArmPlugin::pass::AddArmDequantizeOnInputsGroupConv::AddArmDequantizeOnInputsGroupConv() {
    registerMatcher<opset::ArmGroupConvolution>("AddArmDequantizeOnInputsGroupConv");
}

ArmPlugin::pass::AddArmDequantizeOnInputsAdd::AddArmDequantizeOnInputsAdd() {
    registerMatcher<opset::Add>("AddArmDequantizeOnInputsAdd");
}

ArmPlugin::pass::AddArmDequantizeOnInputsSubtract::AddArmDequantizeOnInputsSubtract() {
    registerMatcher<opset::Subtract>("AddArmDequantizeOnInputsSubtract");
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

bool ArmPlugin::pass::DetectMaybeQuantized::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto reversedOps = f->get_ordered_ops();
    std::reverse(reversedOps.begin(), reversedOps.end());

    for (auto&& node : reversedOps) {
        auto nodeInput = node->inputs();
        auto nodeOutputs = node->outputs();
        ngraph::element::Type type;
        for (auto&& input : nodeInput) {
            if (input.get_element_type().is_quantized()) {
                type = input.get_element_type();
                break;
            }
        }
        if (type == ngraph::element::undefined) {
            for (auto&& output : nodeOutputs) {
                if (output.get_element_type().is_quantized()) {
                    type = output.get_element_type();
                    break;
                }
                auto targetInputs = output.get_target_inputs();
                for (auto&& targetInput : targetInputs) {
                    auto itMaybeQuantized = targetInput.get_node()->get_rt_info().find("MayBeQuanitzed");
                    if (itMaybeQuantized != targetInput.get_node()->get_rt_info().end()) {
                        auto targetType = std::dynamic_pointer_cast<ngraph::VariantWrapper<ngraph::element::Type>>(itMaybeQuantized->second);
                        IE_ASSERT(targetType != nullptr);
                        if (targetType->get() != ngraph::element::undefined) {
                            type = targetType->get();
                            break;
                        }
                    }
                }
                if (type != ngraph::element::undefined) break;
            }
        }
        if (type != ngraph::element::undefined) {
            node->get_rt_info().emplace("MayBeQuanitzed",
                std::make_shared<ngraph::VariantWrapper<ngraph::element::Type>>(type));
        }
    }
    return false;
}

namespace ngraph {
NGRAPH_RTTI_DEFINITION(VariantWrapper<ngraph::element::Type>, "Variant::ngraph::element::Type", 0);
VariantWrapper<ngraph::element::Type>::~VariantWrapper() {}
}  // namespace ngraph