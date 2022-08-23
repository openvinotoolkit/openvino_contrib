// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_algorithm.hpp>

#include "arm_converter/arm_converter.hpp"
#include "opset/opset.hpp"

using namespace InferenceEngine::details;

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
        case ngraph::element::Type_t::i8    : return arm_compute::DataType::S8;
        case ngraph::element::Type_t::i16   : return arm_compute::DataType::S16;
        case ngraph::element::Type_t::u16   : return arm_compute::DataType::U16;
        case ngraph::element::Type_t::i32   : return arm_compute::DataType::S32;
        case ngraph::element::Type_t::u32   : return arm_compute::DataType::U32;
        case ngraph::element::Type_t::i64   : return arm_compute::DataType::S64;
        case ngraph::element::Type_t::f16   : return arm_compute::DataType::F16;
        case ngraph::element::Type_t::f32   : return arm_compute::DataType::F32;
        case ngraph::element::Type_t::bf16  : return arm_compute::DataType::BFLOAT16;
        default: IE_THROW() << "Unsupported Data Type " << type;
    }
}

std::size_t AxisCast(const std::size_t axis, const std::size_t shapeSize) {
    return shapeSize - axis - 1;
}

Converter::Converter(const std::shared_ptr<const ov::Model> model, const Configuration& cfg) :
    _model{model}, _cfg{cfg} {
    Register<opset::Parameter>();
    Register<opset::Constant>();
    Register<opset::ArmConvolution>();
    Register<opset::ArmGroupConvolution>();
    Register<opset::AvgPool>();
    Register<opset::MaxPool>();
    Register<opset::Add>();
    Register<opset::Subtract>();
    Register<opset::Multiply>();
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
    Register<opset::ArmTranspose>();
    Register<opset::Softmax>();
    Register<opset::ArmSplit>();
    Register<opset::LRN>();
    Register<opset::Minimum>();
    Register<opset::Maximum>();
    Register<opset::ArmStridedSlice>();
    Register<opset::Negative>();
    Register<opset::Floor>();
    Register<opset::Exp>();
    Register<opset::MatMul>();
    Register<opset::ArmMatMulBias>();
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
    Register<opset::ArmInterpolate>();
    Register<opset::ArmMVN>();
    Register<opset::ArmNormalizeL2>();
    Register<opset::DepthToSpace>();
    Register<opset::SpaceToDepth>();
    Register<opset::Equal>();
    Register<opset::NotEqual>();
    Register<opset::Less>();
    Register<opset::LessEqual>();
    Register<opset::Greater>();
    Register<opset::GreaterEqual>();
    Register<opset::Select>();
    Register<opset::ReorgYolo>();
    Register<opset::BatchToSpace>();
    Register<opset::SpaceToBatch>();
    Register<opset::ArmConvert>();
    Register<opset::ArmConcat>();
    Register<opset::ArmGather>();
    Register<opset::ArmFFT>();
    Register<opset::ArmQuantize>();
    Register<opset::ArmDequantize>();
    if (_cfg._ref) {
        Register<opset::MVN>();
        Register<opset::NormalizeL2>();
        Register<opset::Interpolate>();
        Register<opset::Concat>();
        Register<opset::Transpose>();
        Register<opset::StridedSlice>();
        Register<opset::Gather>();
        Register<ngraph::op::v1::Gather>();
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
        Register<ngraph::op::v8::DetectionOutput>();
        Register<opset::ReverseSequence>();
        Register<opset::ConvolutionBackpropData>();
        Register<opset::CumSum>();
        Register<opset::FloorMod>();
        Register<opset::CTCGreedyDecoder>();
        Register<opset::CTCGreedyDecoderSeqLen>();
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
        Register<opset::GroupConvolutionBackpropData>();
        Register<opset::OneHot>();
        Register<opset::GatherElements>();
        Register<opset::ReduceLogicalAnd>();
        Register<opset::ReduceLogicalOr>();
        Register<opset::LSTMSequence>();
        Register<opset::GRUSequence>();
        Register<opset::RNNSequence>();
        Register<opset::Bucketize>();
        Register<opset::DFT>();
        Register<opset::IDFT>();
        Register<opset::FakeQuantize>();
        Register<opset::Split>();
        Register<ngraph::op::v8::AdaptiveAvgPool>();
        Register<ngraph::op::v8::AdaptiveMaxPool>();
        Register<ngraph::op::v8::NV12toBGR>();
        Register<ngraph::op::v8::NV12toRGB>();
        Register<ngraph::op::v8::I420toBGR>();
        Register<ngraph::op::v8::I420toRGB>();
        Register<ngraph::op::v8::MaxPool>();
    }
    Register<opset::Result>();
    for (auto&& node : model->get_ordered_ops()) {
        auto& layer = _layers[node->get_instance_id()];
        for (auto&& input : node->inputs()) {
            auto sourceOutput = input.get_source_output();
            layer._inputs.emplace(input, &(_layers.at(sourceOutput.get_node()->get_instance_id())._outputs.at(sourceOutput)));
        }
        if (!ngraph::op::is_output(node)) {
            for (auto&& output : node->outputs()) {
                std::unique_ptr<arm_compute::Tensor> tensor(new arm_compute::Tensor);
                auto tensorShape = ShapeCast(output.get_partial_shape().get_max_shape());
                auto outputDataType = output.get_element_type();
                auto quantizedOutput = (outputDataType == ngraph::element::u8 || outputDataType == ngraph::element::i8);
                arm_compute::TensorInfo tensorInfo;
                if (quantizedOutput && _cfg._lpt) {
                    arm_compute::DataType dataType;
                    switch (outputDataType) {
                        case ngraph::element::Type_t::u8 : dataType = arm_compute::DataType::QASYMM8; break;
                        case ngraph::element::Type_t::i8 : dataType = arm_compute::DataType::QASYMM8_SIGNED; break;
                        default: IE_THROW() << "Arm Plugin: Unsupported Data Type: " << outputDataType << " " << *node;
                    }
                    tensorInfo = {tensorShape, 1, dataType, arm_compute::QuantizationInfo{1, 0}};
                } else {
                    tensorInfo = {tensorShape, 1, DataTypeCast(output.get_element_type())};
                }
                tensor->allocator()->init(tensorInfo);
                layer._outputs.emplace(output, Tensor{std::move(tensor)});
            }
        }
    }
}

Layer::Map Converter::Configure(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager,
                                arm_compute::MemoryGroup& memoryGroup) {
    auto orderedOps = _model->get_ordered_ops();
    std::string unsupported;
    for (auto&& node : orderedOps) {
        if (!contains(_conversions, node->get_type_info())) {
            unsupported += ("\t" + node->get_friendly_name() + " (" + node->get_type_name() + '.' + std::to_string(node->get_type_info().version) + ")\n");
        }
    }
    if (!unsupported.empty()) {
        IE_THROW() << "Arm Plugin: Nodes from " << _model->get_friendly_name() << " are not supported by plugin:\n" << unsupported;
    }
    for (const auto& node : orderedOps) {
        Conversion::Ptr conversion;
        try {
            conversion = _conversions.at(node->get_type_info())(*node);
        } catch(std::exception& e) {
            unsupported += ("\t" + node->get_friendly_name() +
                " (" + node->get_type_name() + '.' + std::to_string(node->get_type_info().version) + ")- " + e.what() + ";\n");
        }
        if (conversion != nullptr) {
            auto status = conversion->Validate();
            if (status.error_code() != arm_compute::ErrorCode::OK) {
                unsupported += ("\t" + node->get_friendly_name() +
                    " (" + node->get_type_name() + '.' + std::to_string(node->get_type_info().version) + ")- " + status.error_description() + ";\n");
            }
        }
    }
    if (!unsupported.empty()) {
        IE_THROW() << "Arm Plugin: Nodes from " << _model->get_friendly_name() << " are not supported:\n" << unsupported;
    }
    std::map<ngraph::Output<ngraph::Node>, std::size_t> counter;
    for (auto&& node : orderedOps) {
        const auto& nodeID = node->get_instance_id();
        if (ngraph::op::is_constant(node)) {
            auto constNode = safe_cast<opset::Constant>(node);
            _layers.at(nodeID)._outputs.begin()->second._tensor->allocator()->import_memory(const_cast<void*>(constNode->get_data_ptr()));
        } else if (!ngraph::op::is_parameter(node) && !ngraph::op::is_output(node)) {
            auto conversion = _conversions.at(node->get_type_info())(*node);
            for (auto&& output : node->outputs()) {
                auto targetInputs = output.get_target_inputs();
                bool isNetworkOutput = std::any_of(std::begin(targetInputs), std::end(targetInputs), [] (auto& targetInput) {
                    return ngraph::op::is_output(targetInput.get_node());
                });
                if (!isNetworkOutput) {
                    counter.emplace(output, targetInputs.size());
                    memoryGroup.manage(_layers.at(nodeID)._outputs.at(output)._tensor.get());
                }
            }
            if (conversion != nullptr) {
                _layers.at(nodeID)._execType = conversion->ExecType();
                conversion->Configure(memoryManager);
            }

            for (auto&& input : node->inputs()) {
                auto tensor = _layers.at(input.get_node()->get_instance_id())._inputs.at(input);
                if (tensor->_tensor->info()->has_padding() && (tensor->_notPaddedTensor != nullptr)) {
                    tensor->_notPaddedTensor->allocator()->init({tensor->_tensor->info()->tensor_shape(), 1, tensor->_tensor->info()->data_type()});
                    memoryGroup.manage(tensor->_notPaddedTensor.get());
                }
            }
            for (auto&& input : node->inputs()) {
                auto tensor = _layers.at(input.get_node()->get_instance_id())._inputs.at(input);
                auto itCounter = counter.find(input.get_source_output());
                if (itCounter != counter.end()) {
                    if ((--(itCounter->second)) == 0) {
                        tensor->_tensor->allocator()->allocate();
                        if (tensor->_tensor->info()->has_padding() && (tensor->_notPaddedTensor != nullptr)) {
                            tensor->_notPaddedTensor->allocator()->allocate();
                        }
                        counter.erase(itCounter);
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
