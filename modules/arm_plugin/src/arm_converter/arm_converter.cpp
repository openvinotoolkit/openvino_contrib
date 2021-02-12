// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>
#include <ie_algorithm.hpp>

#include "arm_converter/arm_converter.hpp"
#include "opset/opset.hpp"


using namespace InferenceEngine::details;

namespace ngraph {
namespace element {
    template <>
    Type from<half_float::half>() {
        return from<ngraph::float16>();
    }
}  //  namespace element
}  //  namespace ngraph

namespace ArmPlugin {
arm_compute::TensorShape ShapeCast(const ngraph::Shape& shape) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        tensorShape.set(shape.size() - i - 1, shape[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

arm_compute::DataType DataTypeCast(const ngraph::element::Type type) {
    switch (static_cast<ngraph::element::Type_t>(type)) {
        case ngraph::element::Type_t::u8    : return arm_compute::DataType::U8;
        case ngraph::element::Type_t::i16   : return arm_compute::DataType::S16;
        case ngraph::element::Type_t::u16   : return arm_compute::DataType::U16;
        case ngraph::element::Type_t::i32   : return arm_compute::DataType::S32;
        case ngraph::element::Type_t::i64   : return arm_compute::DataType::S64;
        case ngraph::element::Type_t::f16   : return arm_compute::DataType::F16;
        case ngraph::element::Type_t::f32   : return arm_compute::DataType::F32;
        case ngraph::element::Type_t::bf16  : return arm_compute::DataType::BFLOAT16;
        default: THROW_IE_EXCEPTION << "Unsupported Data Type " << type; return {};
    }
}

std::size_t AxisCast(const std::size_t axis, const std::size_t shapeSize) {
    return shapeSize - axis - 1;
}

Converter::Converter(const std::shared_ptr<const ngraph::Function> function, bool ref) :
    _function{function} {
    Register<opset::Parameter>();
    Register<opset::Constant>();
    Register<opset::ArmConvolution>();
    Register<opset::ArmGroupConvolution>();
    Register<opset::AvgPool>();
    Register<opset::MaxPool>();
    Register<opset::Add>();
    Register<opset::Subtract>();
    Register<opset::Multiply>();
    Register<opset::Concat>();
    Register<opset::Reshape>();
    Register<opset::Squeeze>();
    Register<opset::Unsqueeze>();
    Register<opset::Sigmoid>();
    Register<opset::Tanh>();
    Register<opset::Relu>();
    Register<opset::PRelu>();
    Register<opset::Abs>();
    Register<opset::Clamp>();
    Register<opset::Sqrt>();
    Register<opset::Elu>();
    Register<opset::Transpose>();
    Register<opset::Softmax>();
    Register<opset::Split>();
    Register<opset::LRN>();
    Register<opset::Minimum>();
    Register<opset::Maximum>();
    Register<opset::StridedSlice>();
    Register<opset::Negative>();
    Register<opset::Floor>();
    Register<opset::Exp>();
    Register<opset::MatMul>();
    Register<opset::MatMulBias>();
    Register<opset::Pad>();
    Register<opset::BatchNormInference>();
    Register<opset::HSwish>();
    Register<opset::SoftPlus>();
    Register<opset::Log>();
    Register<opset::Sin>();
    Register<opset::ShuffleChannels>();
    Register<opset::Power>();
    Register<opset::SquaredDifference>();
    Register<opset::ReduceMean>();
    Register<opset::ReduceSum>();
    Register<opset::ReduceProd>();
    Register<opset::ReduceMin>();
    Register<opset::ReduceMax>();
    Register<opset::Interpolate>();
    Register<opset::MVN>();
    Register<opset::NormalizeL2>();
    Register<opset::DepthToSpace>();
    Register<opset::SpaceToDepth>();
    Register<opset::Equal>();
    Register<opset::NotEqual>();
    Register<opset::Less>();
    Register<opset::LessEqual>();
    Register<opset::Greater>();
    Register<opset::GreaterEqual>();
    Register<opset::Select>();
    Register<opset::Gather>();
    Register<opset::ReorgYolo>();
    Register<opset::BatchToSpace>();
    Register<opset::SpaceToBatch>();
    if (ref) {
        Register<opset::ROIPooling>();
        Register<opset::PSROIPooling>();
        Register<opset::TopK>();
        Register<opset::RegionYolo>();
        Register<opset::Acos>();
        Register<opset::Acosh>();
        Register<opset::Cos>();
        Register<opset::Cosh>();
        Register<opset::Asin>();
        Register<opset::Asinh>();
        Register<opset::Sinh>();
        Register<opset::Atan>();
        Register<opset::Atanh>();
        Register<opset::Tan>();
        Register<opset::Erf>();
        Register<opset::HSigmoid>();
        Register<opset::HardSigmoid>();
        Register<opset::Gelu>();
        Register<opset::Selu>();
        Register<opset::DetectionOutput>();
        Register<opset::ReverseSequence>();
        Register<opset::ConvolutionBackpropData>();
        Register<opset::CumSum>();
        Register<opset::FloorMod>();
        Register<opset::CTCGreedyDecoder>();
        Register<opset::CTCLoss>();
        Register<opset::Round>();
        Register<opset::Convert>();
        Register<opset::ConvertLike>();
        Register<opset::GatherND>();
        Register<opset::ScatterUpdate>();
        Register<opset::ScatterNDUpdate>();
        Register<opset::ScatterElementsUpdate>();
        Register<opset::GatherTree>();
        Register<opset::EmbeddingSegmentsSum>();
        Register<opset::EmbeddingBagPackedSum>();
        Register<opset::EmbeddingBagOffsetsSum>();
        Register<opset::NonMaxSuppression>();
        Register<opset::ROIAlign>();
        Register<ngraph::op::v0::Proposal>();
        Register<opset::Proposal>();
    }
    Register<opset::Result>();

    for (auto&& node : function->get_ordered_ops()) {
        auto& layer = _layers[node->get_friendly_name()];
        for (auto sourceOutput : node->input_values()) {
            layer._inputs.emplace_back(_layers.at(sourceOutput.get_node()->get_friendly_name())._outputs.at(sourceOutput.get_index()).get());
        }
        for (auto output : node->outputs()) {
            std::unique_ptr<arm_compute::Tensor> tensor(new arm_compute::Tensor);
            tensor->allocator()->init({ShapeCast(output.get_partial_shape().get_max_shape()), 1, DataTypeCast(output.get_element_type())});
            layer._outputs.emplace_back(std::move(tensor));
        }
    }
}

Layer::Map Converter::Configure(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager,
                                arm_compute::MemoryGroup& memoryGroup) {
    auto orderedOps = _function->get_ordered_ops();
    std::string unsupported;
    for (auto&& node : orderedOps) {
        if (!contains(_conversions, node->get_type_info())) {
            unsupported += (node->get_friendly_name() + " (" + node->get_type_name() + '.' + std::to_string(node->get_type_info().version) + ") ");
        }
    }
    if (!unsupported.empty()) {
        THROW_IE_EXCEPTION << "Arm Plugin: Nodes from " << _function->get_friendly_name() << " are not supported by plugin: " << unsupported;
    }
    for (const auto& node : orderedOps) {
        Conversion::Ptr conversion;
        try {
            conversion = _conversions.at(node->get_type_info())(*node);
        } catch(std::exception& e) {
            unsupported += (node->get_friendly_name() + " (" + node->get_type_name() + ")- " + e.what() + ";");
        }
        if (conversion != nullptr) {
            auto status = conversion->Validate();
            if (status.error_code() != arm_compute::ErrorCode::OK) {
                unsupported += (node->get_friendly_name() + " (" + node->get_type_name() + ")- " + status.error_description() + ";");
            }
        }
    }
    if (!unsupported.empty()) {
        THROW_IE_EXCEPTION << "Arm Plugin: Nodes from " << _function->get_friendly_name() << " are not supported: " << unsupported;
    }
    std::map<ngraph::Output<ngraph::Node>, std::size_t> counter;
    for (auto&& node : orderedOps) {
        const auto& nodeName = node->get_friendly_name();
        if (ngraph::op::is_constant(node)) {
            auto constNode = std::dynamic_pointer_cast<opset::Constant>(node);
            _layers.at(nodeName)._outputs.at(0)->allocator()->import_memory(const_cast<void*>(constNode->get_data_ptr()));
        } else if (!ngraph::op::is_parameter(node) && !ngraph::op::is_output(node)) {
            auto conversion = _conversions.at(node->get_type_info())(*node);
            for (auto&& output : node->outputs()) {
                auto targetInputs = output.get_target_inputs();
                bool isNetworkOutput = std::any_of(std::begin(targetInputs), std::end(targetInputs), [] (auto& targetInput) {
                    return ngraph::op::is_output(targetInput.get_node());
                });
                if (!isNetworkOutput) {
                    counter.emplace(output, targetInputs.size());
                    memoryGroup.manage(_layers.at(nodeName)._outputs.at(output.get_index()).get());
                }
            }
            if (conversion != nullptr) {
                conversion->Configure(memoryManager);
            }
            for (auto&& input : node->inputs()) {
                auto itCounter = counter.find(input.get_source_output());
                if (itCounter != counter.end()) {
                    if ((--(itCounter->second)) == 0) {
                        _layers.at(nodeName)._inputs.at(input.get_index())->allocator()->allocate();
                    }
                }
            }
        }
    }
    return std::move(_layers);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Parameter& node) {
    return {};
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Result& node) {
    return {};
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Constant& node) {
    return {};
}
}  //  namespace ArmPlugin
