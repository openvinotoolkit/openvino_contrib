// Copyright (C) 2020-2022 Intel Corporation
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
std::vector<float> getFloatVector(const ngraph::Node* constant) {
    auto outputType = constant->get_output_element_type(0);
    if (outputType == ngraph::element::f32) {
        return safe_cast<const opset::Constant>(constant)->cast_vector<float>();
    } else if (outputType == ngraph::element::f16) {
        auto vec = safe_cast<const opset::Constant>(constant)->cast_vector<ngraph::float16>();
        return {std::begin(vec), std::end(vec)};
    } else if (outputType == ngraph::element::i8) {
        auto vec = safe_cast<const opset::Constant>(constant)->cast_vector<std::int8_t>();
        return {std::begin(vec), std::end(vec)};
    } else if (outputType == ngraph::element::u8) {
        auto vec = safe_cast<const opset::Constant>(constant)->cast_vector<std::uint8_t>();
        return {std::begin(vec), std::end(vec)};
    } else {
        IE_THROW() << "Unsupported element type: " << outputType;
    }
}

std::pair<std::vector<float>, std::vector<float>> makeQuantizationInfo(const std::vector<float>& input_low, const std::vector<float>& input_high,
                                                                       const std::vector<float>& output_low, const std::vector<float>& output_high) {
    IE_ASSERT(input_low.size() == input_high.size());
    std::pair<std::vector<float>, std::vector<float>> qVector;
    for (std::size_t i = 0; i < input_low.size(); ++i) {
        auto scale = (output_high[0] - output_low[0]) / (input_high[i] - input_low[i]);
        qVector.first.emplace_back(scale);
        qVector.second.emplace_back(output_low[0] - input_low[i] * scale);
    }
    return qVector;
}

template <typename T>
bool allEqualToFirst(const T& values) {
    auto& ref = *std::begin(values);
    return std::all_of(std::begin(values), std::end(values), [&] (auto& value) {return value == ref;});
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
    IE_ASSERT(!"Arm Plugin: Unregistered type: ") << node;
#undef CASE
}
}  // namespace

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertQuantize, "ConvertQuantize", 0);
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
            auto fakeQuantize = safe_cast<opset::FakeQuantize>(m.get_match_root());
            auto input_type = fakeQuantize->input(0).get_element_type();
            auto output_type = fakeQuantize->output(0).get_element_type();
            auto input = fakeQuantize->input_value(0);
            auto input_low = getFloatVector(fakeQuantize->input_value(1).get_node());
            auto input_high = getFloatVector(fakeQuantize->input_value(2).get_node());
            auto output_low = getFloatVector(fakeQuantize->input_value(3).get_node());
            auto output_high = getFloatVector(fakeQuantize->input_value(4).get_node());
            using Types = std::vector<ngraph::element::Type>;
            if ((input_type.is_real() || input_type.is_quantized()) && output_type.is_quantized()) {
                auto qInfo = makeQuantizationInfo(input_low, input_high, output_low, output_high);

                std::shared_ptr<ngraph::op::TypeRelaxed<opset::ArmQuantize>> armQuantize;
                if (qInfo.first.size() > 1) {
                    auto fInput = input;
                    if (input_type.is_quantized()) {
                        auto dqNode = std::make_shared<opset::ArmDequantize>(input);
                        OPENVINO_ASSERT(dqNode, "Failed to create ArmDequantize node for per channel requantization");
                        dqNode->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_dequantize_prescale");
                        ngraph::copy_runtime_info(fakeQuantize, dqNode);
                        dqNode->get_rt_info()["QuantizationInfo"] = arm_compute::QuantizationInfo{1, 0};
                        fInput = dqNode;
                    }

                    auto quantScale = opset::Constant::create<float>(input_type, fakeQuantize->get_input_shape(1), qInfo.first);
                    auto quantMultiply = std::make_shared<opset::Multiply>(fInput, quantScale);
                    quantMultiply->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize_scale");
                    ngraph::copy_runtime_info(fakeQuantize, quantMultiply);

                    auto quantShift = opset::Constant::create<float>(input_type, fakeQuantize->get_input_shape(1), qInfo.second);
                    auto quantAdd = std::make_shared<opset::Add>(quantMultiply, quantShift);
                    quantAdd->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize_shift");
                    ngraph::copy_runtime_info(fakeQuantize, quantAdd);

                    armQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmQuantize>>(Types{input_type}, Types{output_type}, quantAdd);
                    ngraph::copy_runtime_info(fakeQuantize, armQuantize);
                    armQuantize->get_rt_info()["QuantizationInfo"] = arm_compute::QuantizationInfo{1, 0};
                } else {
                    armQuantize = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmQuantize>>(Types{input_type}, Types{output_type}, input);
                    ngraph::copy_runtime_info(fakeQuantize, armQuantize);
                    armQuantize->get_rt_info()["QuantizationInfo"] =
                        arm_compute::QuantizationInfo{1.f/qInfo.first[0], static_cast<std::int32_t>(std::round(qInfo.second[0]))};
                }
                armQuantize->set_friendly_name(fakeQuantize->get_friendly_name() + "_arm_quantize");
                ngraph::replace_node(fakeQuantize, armQuantize);

                return true;
            }
            return false;
        });
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvolutionQuantizeFusion, "ConvolutionQuantizeFusion", 0);
ArmPlugin::pass::ConvolutionQuantizeFusion::ConvolutionQuantizeFusion() {
    auto node_pattern = ngraph::pattern::wrap_type<
        opset::ArmConvolution,
        opset::ArmGroupConvolution,
        opset::MatMul>(ngraph::pattern::consumers_count(1));
    auto activation_pattern = ngraph::pattern::wrap_type<
        opset::Sigmoid, opset::Tanh, opset::Relu, opset::Abs,
        opset::Elu, opset::Sqrt, opset::SoftPlus, opset::HSwish,
        opset::PRelu, opset::Clamp>({node_pattern});
    auto q_scale = ngraph::pattern::wrap_type<opset::Constant>();
    auto q_mul = ngraph::pattern::wrap_type<opset::Multiply>({
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{node_pattern, activation_pattern}),
        q_scale},
        ngraph::pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{node_pattern, activation_pattern, q_mul}),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fq_pattern, "ConvolutionQuantizeFusion"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto node = pattern_map[node_pattern].get_node_shared_ptr();
            auto fakeQuantize = safe_cast<opset::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
            auto itMul = pattern_map.find(q_mul);
            auto itActivation = pattern_map.find(activation_pattern);
            auto realType = node->get_output_element_type(0);
            auto quantizedType = fakeQuantize->get_output_element_type(0);
            if (!(realType.is_real() && quantizedType.is_quantized())) {
                return false;
            }
            auto quantizationInfo = makeQuantizationInfo(getFloatVector(fakeQuantize->input_value(1).get_node()),
                                                         getFloatVector(fakeQuantize->input_value(2).get_node()),
                                                         getFloatVector(fakeQuantize->input_value(3).get_node()),
                                                         getFloatVector(fakeQuantize->input_value(4).get_node()));
            if (itMul != pattern_map.end()) {
                std::vector<float> scales = getFloatVector(pattern_map[q_scale].get_node());
                if (allEqualToFirst(scales)) {
                    if (scales[0] == 0.) return false; // Scale multiplier shouldn't be zero
                    for (auto&& v : quantizationInfo.first) v *= scales[0];
                } else if (quantizationInfo.first.size() > 1) {
                    if (scales.size() != quantizationInfo.first.size()) return false;
                    std::transform(quantizationInfo.first.begin(), quantizationInfo.first.end(), scales.begin(),
                                   quantizationInfo.first.begin(), [](float f, float sc) -> float { return f * sc; });
                } else {
                    std::vector<float> qiScales, qiOffsets;
                    for (auto&& sc : scales) {
                        qiScales.emplace_back(quantizationInfo.first[0] * sc);
                        qiOffsets.emplace_back(quantizationInfo.second[0]);
                    }
                    quantizationInfo.first.swap(qiScales);
                    quantizationInfo.second.swap(qiOffsets);
                }
            }

            std::vector<ngraph::Output<ngraph::Node>> newInputs;
            Types inputTypes;
            for (auto&& input : node->inputs()) {
                inputTypes.emplace_back(realType);
                newInputs.emplace_back(
                    ngraph::op::TemporaryReplaceOutputType{input.get_source_output(), realType}.get());
            }

            std::shared_ptr<ngraph::Node> bias;
            if (node->inputs().size() > 2) {
                bias = node->input_value(2).get_node_shared_ptr();
            }

            bool negativeScales = std::any_of(std::begin(quantizationInfo.first), std::end(quantizationInfo.first), [] (auto& value) {return value < 0;});
            if (negativeScales) {
                if (node->get_input_element_type(1) != ngraph::element::i8)
                    return false;
                std::vector<std::int8_t> negate;
                std::transform(quantizationInfo.first.begin(), quantizationInfo.first.end(), std::back_inserter(negate),
                                                                                             [](float f) -> std::int8_t { return f < 0 ? -1 : 1; } );
                std::transform(quantizationInfo.first.begin(), quantizationInfo.first.end(), quantizationInfo.first.begin(),
                                                                                             [](float f) -> float { return f < 0 ? -f : f; } );
                std::shared_ptr<ngraph::Node> weightMultiply;
                if (ngraph::is_type<opset::Constant>(node->input_value(1).get_node())) {
                    std::vector<std::int8_t> weights = safe_cast<const opset::Constant>(node->input_value(1).get_node())->cast_vector<std::int8_t>();
                    size_t step = weights.size() / negate.size();
                    auto weightsIt = weights.begin();
                    for (auto&& sign : negate) {
                        std::transform(weightsIt, weightsIt + step, weightsIt, [&sign](std::int8_t w) -> std::int8_t { return w * sign; } );
                        weightsIt += step;
                    }
                    weightMultiply = std::make_shared<opset::Constant>(node->get_input_element_type(1), node->get_input_shape(1), weights);
                } else {
                    weightMultiply = std::make_shared<opset::Multiply>(node->input_value(1),
                                                                       std::make_shared<opset::Constant>(node->get_input_element_type(1),
                                                                                                         ngraph::Shape{negate.size(), 1, 1, 1}, negate));
                }
                weightMultiply->set_friendly_name(node->input_value(1).get_node_shared_ptr()->get_friendly_name() + "_weights_negate");
                ngraph::copy_runtime_info(node->input_value(1).get_node_shared_ptr(), weightMultiply);
                newInputs[1] = ngraph::op::TemporaryReplaceOutputType{weightMultiply->output(0), ngraph::element::i8}.get();

                if (bias) {
                    bias = std::make_shared<opset::Multiply>(bias,
                                                             std::make_shared<opset::Constant>(ngraph::element::f32,
                                                                                               ngraph::Shape{negate.size()}, negate));
                    bias->set_friendly_name(node->input_value(2).get_node_shared_ptr()->get_friendly_name() + "_bias_negate");
                    ngraph::copy_runtime_info(node->input_value(2).get_node_shared_ptr(), bias);
                    newInputs[2] = ngraph::op::TemporaryReplaceOutputType{bias->output(0), realType}.get();
                }
            }

            std::int32_t qiOffset = 0;
            if (!allEqualToFirst(quantizationInfo.second)) {
                std::transform(quantizationInfo.second.begin(), quantizationInfo.second.end(), quantizationInfo.first.begin(),
                               quantizationInfo.second.begin(), [](float sh, float sc) -> float { return sh / sc; } );
                std::shared_ptr<ngraph::Node> zpbias = std::make_shared<opset::Constant>(ngraph::element::f32,
                                                                                         ngraph::Shape{quantizationInfo.second.size()},
                                                                                         quantizationInfo.second);
                OPENVINO_ASSERT(zpbias, "Failed to convert zero point to bias node for fused convolution");
                if (bias) {
                    bias = std::make_shared<opset::Add>(bias, zpbias);
                    bias->set_friendly_name(node->input_value(2).get_node_shared_ptr()->get_friendly_name() + "_bias_fusedzp");
                    ngraph::copy_runtime_info(node->input_value(2).get_node_shared_ptr(), bias);
                    newInputs[2] = ngraph::op::TemporaryReplaceOutputType{bias->output(0), realType}.get();
                } else {
                    inputTypes.emplace_back(realType);
                    newInputs.emplace_back(ngraph::op::TemporaryReplaceOutputType{zpbias->output(0), realType}.get());
                }
            } else {
                qiOffset = static_cast<std::int32_t>(std::round(quantizationInfo.second[0]));
            }
            auto newNode = makeTypeRelaxed(node.get(), newInputs, inputTypes, Types{quantizedType});
            if (!bias && newInputs.size() == 3 && newNode->inputs().size() != 3) {
                //TypeRelaxed operations unable to extend amount of inputs on copy
                newNode->set_argument(2, newInputs.at(2));
            }

            if (itActivation != pattern_map.end()) {
                auto activation = itActivation->second.get_node_shared_ptr();
                auto activationLayerInfo = opset::makeActivationLayerInfo(activation.get());
                newNode->set_friendly_name(node->get_friendly_name() + '_' +
                                           activation->get_friendly_name() + '_' +
                                           fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info({node, activation, fakeQuantize}, newNode);
                if (activationLayerInfo.enabled()) {
                    newNode->get_rt_info().emplace("ActivationLayerInfo", activationLayerInfo);
                }
            } else {
                newNode->set_friendly_name(node->get_friendly_name() + '_' + fakeQuantize->get_friendly_name());
                ngraph::copy_runtime_info({node, fakeQuantize}, newNode);
            }

            float qiScale = 1.f;
            if (!allEqualToFirst(quantizationInfo.first)) {
                if (node->get_input_element_type(1) != ngraph::element::i8)
                    return false;
                //Is it correct if fused activation exists?
                newNode->get_rt_info()["WeightsPrescaleInfo"] =
                    arm_compute::QuantizationInfo{quantizationInfo.first, std::vector<std::int32_t>(quantizationInfo.first.size(), 0)};
            } else {
                qiScale = 1.f/quantizationInfo.first[0];
            }

            newNode->get_rt_info()["QuantizationInfo"] =
                arm_compute::QuantizationInfo{qiScale, qiOffset};

            ngraph::replace_node(fakeQuantize, newNode);
            return true;
        });
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::MeanQuantizeFusion, "MeanQuantizeFusion", 0);
ArmPlugin::pass::MeanQuantizeFusion::MeanQuantizeFusion() {
    auto node_pattern = ngraph::pattern::wrap_type<
        opset::AvgPool,
        opset::ReduceMean>(ngraph::pattern::consumers_count(1));
    auto fq_pattern = ngraph::pattern::wrap_type<opset::FakeQuantize>({
        node_pattern,
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
        ngraph::pattern::has_static_shape());
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(fq_pattern, "MeanQuantizeFusion"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto node = pattern_map[node_pattern].get_node_shared_ptr();
            if (ngraph::is_type<opset::AvgPool>(node) && node->get_input_shape(0).size() == 5) {
                return false;
            }
            auto fakeQuantize = safe_cast<opset::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
            auto realType = node->get_output_element_type(0);
            auto quantizedType = fakeQuantize->get_output_element_type(0);
            if (!(realType.is_real() && quantizedType.is_quantized())) {
                return false;
            }

            auto quantizationInfo = makeQuantizationInfo(getFloatVector(fakeQuantize->input_value(1).get_node()),
                                                         getFloatVector(fakeQuantize->input_value(2).get_node()),
                                                         getFloatVector(fakeQuantize->input_value(3).get_node()),
                                                         getFloatVector(fakeQuantize->input_value(4).get_node()));
            if (!allEqualToFirst(quantizationInfo.first) || !allEqualToFirst(quantizationInfo.second)) {
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
            newNode->set_friendly_name(node->get_friendly_name() + '_' + fakeQuantize->get_friendly_name());
            ngraph::copy_runtime_info({node, fakeQuantize}, newNode);
            newNode->get_rt_info()["QuantizationInfo"] =
                arm_compute::QuantizationInfo{1.f/quantizationInfo.first[0], static_cast<std::int32_t>(std::round(quantizationInfo.second[0]))};
            ngraph::replace_node(fakeQuantize, newNode);
            return true;
        });
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::DequantizeInputFusion, "DequantizeInputFusion", 0);
ArmPlugin::pass::DequantizeInputFusion::DequantizeInputFusion() {
    auto scale_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset::Multiply>(
                           {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                           scale_pattern},
                           ngraph::pattern::consumers_count(1));

    auto offset_pattern = ngraph::pattern::wrap_type<opset::Constant>();
    auto add_pattern = ngraph::pattern::wrap_type<opset::Add>(
                           {std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{
                               mul_pattern,
                               ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                               }),
                           offset_pattern},
                           ngraph::pattern::consumers_count(1));
    auto sub_pattern = ngraph::pattern::wrap_type<opset::Subtract>(
                           {std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{
                               mul_pattern,
                               ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                               }),
                           offset_pattern},
                           ngraph::pattern::consumers_count(1));

    auto preadd_pattern = ngraph::pattern::wrap_type<opset::Add>(
                              {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                              offset_pattern},
                              ngraph::pattern::consumers_count(1));
    auto presub_pattern = ngraph::pattern::wrap_type<opset::Subtract>(
                              {ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                              offset_pattern},
                              ngraph::pattern::consumers_count(1));
    auto postmul_pattern = ngraph::pattern::wrap_type<opset::Multiply>(
                               {std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{
                                   preadd_pattern,
                                   presub_pattern,
                                   }),
                               scale_pattern},
                               ngraph::pattern::consumers_count(1));

    auto dequantize_output = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{
        postmul_pattern, add_pattern, sub_pattern, mul_pattern});
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(dequantize_output, "DequantizeInputFusion"),
        [=](ngraph::pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_value_map();
            auto itPostMul = pattern_map.find(postmul_pattern);
            auto itPreAdd = pattern_map.find(preadd_pattern);
            auto itPreSub = pattern_map.find(presub_pattern);
            auto itAdd = pattern_map.find(add_pattern);
            auto itSub = pattern_map.find(sub_pattern);
            auto itMul = pattern_map.find(mul_pattern);

            auto output = [&] {
                for (auto& it : {itPostMul, itAdd, itSub, itMul}) if (it != pattern_map.end()) return it->second.get_node_shared_ptr();
                IE_ASSERT(!"Arm Plugin: No output pattern found!");
            }();
            auto realType = output->get_output_element_type(0);
            if (output->output(0).get_target_inputs().size() != 1 || !realType.is_real()) {
                return false;
            }

            auto input = [&] {
                for (auto& it : {itPreAdd, itPreSub, itMul, itAdd, itSub}) if (it != pattern_map.end()) return it->second.get_node_shared_ptr();
                IE_ASSERT(!"Arm Plugin: No input pattern found!");
            }();
            auto quantizedType = input->get_input_element_type(0);
            if (quantizedType != ngraph::element::i8 && quantizedType != ngraph::element::u8) {
                return false;
            }

            auto node = output->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
            if (ngraph::is_type<opset::AvgPool>(node) && (node->get_input_shape(0).size() == 5) &&
                node->get_output_element_type(0).is_quantized()) {
                    return false;
                }

            float scale = 1.f;
            std::int32_t offset = 0;
            if (itPostMul != pattern_map.end() || itMul != pattern_map.end()) {
                std::vector<float> scales = getFloatVector(pattern_map[scale_pattern].get_node());
                if (!allEqualToFirst(scales)) return false;
                scale = scales.front();
            }
            if (itPreAdd != pattern_map.end() || itPreSub != pattern_map.end() ||
                itAdd != pattern_map.end() || itSub != pattern_map.end()) {
                std::vector<float> offsets = getFloatVector(pattern_map[offset_pattern].get_node());
                if (!allEqualToFirst(offsets)) return false;
                float foffset = (itPreAdd != pattern_map.end() || itAdd != pattern_map.end()) ? - offsets.front() : offsets.front();
                if (itMul != pattern_map.end()) foffset /= scale;
                offset = static_cast<std::int32_t>(std::round(itPreAdd != pattern_map.end() ? - offsets.front() :
                                                             (itAdd != pattern_map.end() ? - offsets.front() : offsets.front()) / scale));
            }

            std::vector<std::shared_ptr<ngraph::Node>> nodesToCopyRTI;
            for (auto& pattern : {scale_pattern, mul_pattern,
                                  offset_pattern, add_pattern, sub_pattern,
                                  preadd_pattern, presub_pattern, postmul_pattern}) {
                auto itPattern = pattern_map.find(pattern);
                if (itPattern != pattern_map.end()) {
                    nodesToCopyRTI.emplace_back(itPattern->second.get_node_shared_ptr());
                }
            }

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
                newInputs[0] = ngraph::op::TemporaryReplaceOutputType{input->input_value(0), realType}.get();
                newNode = makeTypeRelaxed(node.get(), newInputs, inputTypes, Types{node->get_output_element_type(0)});
                newNode->set_friendly_name(node->get_friendly_name());
                nodesToCopyRTI.emplace_back(node);
                ngraph::copy_runtime_info(nodesToCopyRTI, newNode);
                newNode->get_rt_info()["InputPrescaleInfo"] = arm_compute::QuantizationInfo(scale, offset);
                nodeToReplace = node;
            } else {
                newNode = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(Types{quantizedType}, Types{realType},
                                                                                          input->input_value(0));
                newNode->set_friendly_name(output->get_friendly_name() + "_arm_dequantize");
                ngraph::copy_runtime_info(nodesToCopyRTI, newNode);
                newNode->get_rt_info()["QuantizationInfo"] = arm_compute::QuantizationInfo(scale, offset);
                nodeToReplace = output;
            }
            ngraph::replace_node(nodeToReplace, newNode);
            return true;
        });
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::AddDequantizeOnInputs, "AddDequantizeOnInputs", 0);
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
                        if ((inputType != ngraph::element::i8 && inputType != ngraph::element::u8) ||
                            ngraph::op::is_constant(input.get_source_output().get_node())) {
                            newInputOp = std::make_shared<opset::Convert>(input.get_source_output(), outputType);
                        } else {
                            newInputOp = std::make_shared<ngraph::op::TypeRelaxed<opset::ArmDequantize>>(Types{inputType}, Types{outputType},
                                                                                                         input.get_source_output());
                            newInputOp->set_friendly_name(node->get_friendly_name() + "_on_input_" + std::to_string(input.get_index()) + "_arm_dequantize");
                            newInputOp->get_rt_info()["QuantizationInfo"] = arm_compute::QuantizationInfo{1, 0};
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

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertBiasToI32, "ConvertBiasToI32", 0);
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
