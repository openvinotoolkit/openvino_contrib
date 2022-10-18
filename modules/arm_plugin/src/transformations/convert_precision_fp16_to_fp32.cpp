// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_precision_fp16_to_fp32.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset9.hpp>
#include <openvino/opsets/opset3.hpp>

#define CHECK_TYPE(TYPE, node)                                             \
    if (ov::is_type<TYPE>(node)) {                                         \
        return false;                                                      \
    }

bool check_op_type(const std::shared_ptr<ov::Node>& op) {
    CHECK_TYPE(ov::opset9::MVN, op);
    CHECK_TYPE(ov::opset9::NormalizeL2, op);
    CHECK_TYPE(ov::opset9::Interpolate, op);
    CHECK_TYPE(ov::opset9::Concat, op);
    CHECK_TYPE(ov::opset9::Transpose, op);
    CHECK_TYPE(ov::opset9::StridedSlice, op);
    CHECK_TYPE(ov::opset9::Gather, op);
    CHECK_TYPE(ov::opset3::Gather, op);
    CHECK_TYPE(ov::opset9::ROIPooling, op);
    CHECK_TYPE(ov::opset9::PSROIPooling, op);
    CHECK_TYPE(ov::opset9::TopK, op);
    CHECK_TYPE(ov::opset9::RegionYolo, op);
    CHECK_TYPE(ov::opset9::Acos, op);
    CHECK_TYPE(ov::opset9::Acosh, op);
    CHECK_TYPE(ov::opset9::Cos, op);
    CHECK_TYPE(ov::opset9::Cosh, op);
    CHECK_TYPE(ov::opset9::Asin, op);
    CHECK_TYPE(ov::opset9::Asinh, op);
    CHECK_TYPE(ov::opset9::Sinh, op);
    CHECK_TYPE(ov::opset9::Atan, op);
    CHECK_TYPE(ov::opset9::Atanh, op);
    CHECK_TYPE(ov::opset9::Tan, op);
    CHECK_TYPE(ov::opset9::Erf, op);
    CHECK_TYPE(ov::opset9::HSigmoid, op);
    CHECK_TYPE(ov::opset9::HardSigmoid, op);
    CHECK_TYPE(ov::opset9::Gelu, op);
    CHECK_TYPE(ov::opset9::Selu, op);
    CHECK_TYPE(ov::opset9::DetectionOutput, op);
    CHECK_TYPE(ov::opset9::DetectionOutput, op);
    CHECK_TYPE(ov::opset9::ReverseSequence, op);
    CHECK_TYPE(ov::opset9::ConvolutionBackpropData, op);
    CHECK_TYPE(ov::opset9::CumSum, op);
    CHECK_TYPE(ov::opset9::FloorMod, op);
    CHECK_TYPE(ov::opset9::CTCGreedyDecoder, op);
    CHECK_TYPE(ov::opset9::CTCGreedyDecoderSeqLen, op);
    CHECK_TYPE(ov::opset9::CTCLoss, op);
    CHECK_TYPE(ov::opset9::Round, op);
    CHECK_TYPE(ov::opset9::Convert, op);
    CHECK_TYPE(ov::opset9::ConvertLike, op);
    CHECK_TYPE(ov::opset9::GatherND, op);
    CHECK_TYPE(ov::opset9::ScatterUpdate, op);
    CHECK_TYPE(ov::opset9::ScatterNDUpdate, op);
    CHECK_TYPE(ov::opset9::ScatterElementsUpdate, op);
    CHECK_TYPE(ov::opset9::GatherTree, op);
    CHECK_TYPE(ov::opset9::EmbeddingSegmentsSum, op);
    CHECK_TYPE(ov::opset9::EmbeddingBagPackedSum, op);
    CHECK_TYPE(ov::opset9::EmbeddingBagOffsetsSum, op);
    CHECK_TYPE(ov::opset9::NonMaxSuppression, op);
    CHECK_TYPE(ov::opset9::ROIAlign, op);
    CHECK_TYPE(ov::opset3::Proposal, op);
    CHECK_TYPE(ov::opset9::Proposal, op);
    CHECK_TYPE(ov::opset9::GroupConvolutionBackpropData, op);
    CHECK_TYPE(ov::opset9::OneHot, op);
    CHECK_TYPE(ov::opset9::GatherElements, op);
    CHECK_TYPE(ov::opset9::ReduceLogicalAnd, op);
    CHECK_TYPE(ov::opset9::ReduceLogicalOr, op);
    CHECK_TYPE(ov::opset9::LSTMSequence, op);
    CHECK_TYPE(ov::opset9::GRUSequence, op);
    CHECK_TYPE(ov::opset9::RNNSequence, op);
    CHECK_TYPE(ov::opset9::Bucketize, op);
    CHECK_TYPE(ov::opset9::DFT, op);
    CHECK_TYPE(ov::opset9::IDFT, op);
    CHECK_TYPE(ov::opset9::FakeQuantize, op);
    CHECK_TYPE(ov::opset9::Split, op);
    CHECK_TYPE(ov::opset9::AdaptiveAvgPool, op);
    CHECK_TYPE(ov::opset9::AdaptiveMaxPool, op);
    CHECK_TYPE(ov::opset9::NV12toBGR, op);
    CHECK_TYPE(ov::opset9::NV12toRGB, op);
    CHECK_TYPE(ov::opset9::I420toBGR, op);
    CHECK_TYPE(ov::opset9::I420toRGB, op);
    CHECK_TYPE(ov::opset9::MaxPool, op);
    return true;
}

bool ArmPlugin::pass::ConvertPrecisionFP16ToFP32::run_on_model(const std::shared_ptr<ov::Model>& m) {
    const auto ordered_ops = m->get_ordered_ops();
    for (const auto& op : ordered_ops) {
        bool is_natively_supported = check_op_type(op);
        if (is_natively_supported) {
            continue;
        }

        bool convert_for_outputs_required = false;
        for (const auto& input : op->inputs()) {
            if (input.get_element_type() == ov::element::f16) {
                auto parent_output = input.get_source_output();
                auto parent_convert = std::dynamic_pointer_cast<ov::opset9::Convert>(parent_output.get_node_shared_ptr());
                if (parent_convert && parent_convert->get_rt_info().count("fp16_to_fp32_transformation") != 0) {
                    input.replace_source_output(parent_convert->input_value(0));
                } else {
                    auto convert = std::make_shared<ov::opset9::Convert>(input.get_source_output(), ov::element::f32);
                    convert->output(0).add_names(parent_output.get_names());
                    input.replace_source_output(convert);
                }
                convert_for_outputs_required = true;
            }
        }

        if (convert_for_outputs_required) {
            // propagate fp32 precision into outputs
            op->validate_and_infer_types();
            for (auto& output : op->outputs()) {
                if (output.get_element_type() == ov::element::f32) {
                    auto target_inputs = output.get_target_inputs();
                    auto convert = std::make_shared<ov::opset9::Convert>(output, ov::element::f16);

                    auto& rt_info = convert->get_rt_info();
                    rt_info["fp16_to_fp32_transformation"] = "";
                    for (const auto& target_input : target_inputs) {
                        target_input.replace_source_output(convert);
                    }

                    convert->output(0).get_tensor_ptr()->add_names(output.get_names());
                }
            }
        }

        auto multisubgraph_op = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(op);
        if (multisubgraph_op) {
            for (size_t idx = 0; idx < multisubgraph_op->get_internal_subgraphs_size(); ++idx) {
                run_on_model(multisubgraph_op->get_function(static_cast<int>(idx)));
            }
        }
    }
    return true;
}
