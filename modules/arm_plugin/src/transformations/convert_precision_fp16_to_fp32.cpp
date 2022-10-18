// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_precision_fp16_to_fp32.hpp"
#include "opset/opset.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset9.hpp>

#define CHECK_TYPE(TYPE, node)                                             \
    if (ov::is_type<TYPE>(node)) {                                         \
        return true;                                                       \
    }

bool check_op_type(const std::shared_ptr<ov::Node>& op) {
    CHECK_TYPE(opset::Parameter, op);
    CHECK_TYPE(opset::Constant, op);
    CHECK_TYPE(opset::ArmConvolution, op);
    CHECK_TYPE(opset::ArmGroupConvolution, op);
    CHECK_TYPE(opset::AvgPool, op);
    CHECK_TYPE(opset::MaxPool, op);
    CHECK_TYPE(opset::Add, op);
    CHECK_TYPE(opset::Subtract, op);
    CHECK_TYPE(opset::Multiply, op);
    CHECK_TYPE(opset::Reshape, op);
    CHECK_TYPE(opset::Squeeze, op);
    CHECK_TYPE(opset::Unsqueeze, op);
    CHECK_TYPE(opset::Sigmoid, op);
    CHECK_TYPE(opset::Tanh, op);
    CHECK_TYPE(opset::Relu, op);
    CHECK_TYPE(opset::PRelu, op);
    CHECK_TYPE(opset::Abs, op);
    CHECK_TYPE(opset::Clamp, op);
    CHECK_TYPE(opset::Sqrt, op);
    CHECK_TYPE(opset::Elu, op);
    CHECK_TYPE(opset::ArmTranspose, op);
    CHECK_TYPE(opset::Softmax, op);
    CHECK_TYPE(opset::ArmSplit, op);
    CHECK_TYPE(opset::LRN, op);
    CHECK_TYPE(opset::Minimum, op);
    CHECK_TYPE(opset::Maximum, op);
    CHECK_TYPE(opset::ArmStridedSlice, op);
    CHECK_TYPE(opset::Negative, op);
    CHECK_TYPE(opset::Floor, op);
    CHECK_TYPE(opset::Exp, op);
    CHECK_TYPE(opset::MatMul, op);
    CHECK_TYPE(opset::ArmMatMulBias, op);
    CHECK_TYPE(opset::Pad, op);
    CHECK_TYPE(opset::BatchNormInference, op);
    CHECK_TYPE(opset::HSwish, op);
    CHECK_TYPE(opset::SoftPlus, op);
    CHECK_TYPE(opset::Log, op);
    CHECK_TYPE(opset::Sin, op);
    CHECK_TYPE(opset::ShuffleChannels, op);
    CHECK_TYPE(opset::Power, op);
    CHECK_TYPE(opset::SquaredDifference, op);
    CHECK_TYPE(opset::ReduceMean, op);
    CHECK_TYPE(opset::ReduceSum, op);
    CHECK_TYPE(opset::ReduceProd, op);
    CHECK_TYPE(opset::ReduceMin, op);
    CHECK_TYPE(opset::ReduceMax, op);
    CHECK_TYPE(opset::ArmInterpolate, op);
    CHECK_TYPE(opset::ArmMVN, op);
    CHECK_TYPE(opset::ArmNormalizeL2, op);
    CHECK_TYPE(opset::DepthToSpace, op);
    CHECK_TYPE(opset::SpaceToDepth, op);
    CHECK_TYPE(opset::Equal, op);
    CHECK_TYPE(opset::NotEqual, op);
    CHECK_TYPE(opset::Less, op);
    CHECK_TYPE(opset::LessEqual, op);
    CHECK_TYPE(opset::Greater, op);
    CHECK_TYPE(opset::GreaterEqual, op);
    CHECK_TYPE(opset::Select, op);
    CHECK_TYPE(opset::ReorgYolo, op);
    CHECK_TYPE(opset::BatchToSpace, op);
    CHECK_TYPE(opset::SpaceToBatch, op);
    CHECK_TYPE(opset::ArmConvert, op);
    CHECK_TYPE(opset::ArmConcat, op);
    CHECK_TYPE(opset::ArmGather, op);
    CHECK_TYPE(opset::ArmFFT, op);
    CHECK_TYPE(opset::ArmQuantize, op);
    CHECK_TYPE(opset::ArmDequantize, op);
    return false;
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
