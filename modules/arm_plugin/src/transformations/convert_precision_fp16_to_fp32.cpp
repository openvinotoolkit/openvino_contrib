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
    CHECK_TYPE(ArmPlugin::opset::Parameter, op);
    CHECK_TYPE(ArmPlugin::opset::Constant, op);
    CHECK_TYPE(ArmPlugin::opset::ArmConvolution, op);
    CHECK_TYPE(ArmPlugin::opset::ArmGroupConvolution, op);
    CHECK_TYPE(ArmPlugin::opset::AvgPool, op);
    CHECK_TYPE(ArmPlugin::opset::MaxPool, op);
    CHECK_TYPE(ArmPlugin::opset::Add, op);
    CHECK_TYPE(ArmPlugin::opset::Subtract, op);
    CHECK_TYPE(ArmPlugin::opset::Multiply, op);
    CHECK_TYPE(ArmPlugin::opset::Reshape, op);
    CHECK_TYPE(ArmPlugin::opset::Squeeze, op);
    CHECK_TYPE(ArmPlugin::opset::Unsqueeze, op);
    CHECK_TYPE(ArmPlugin::opset::Sigmoid, op);
//    Failed native layer in ACL for FP16 precision 
//    CHECK_TYPE(ArmPlugin::opset::Tanh, op);
    CHECK_TYPE(ArmPlugin::opset::Relu, op);
    CHECK_TYPE(ArmPlugin::opset::PRelu, op);
    CHECK_TYPE(ArmPlugin::opset::Abs, op);
    CHECK_TYPE(ArmPlugin::opset::Clamp, op);
    CHECK_TYPE(ArmPlugin::opset::Sqrt, op);
    CHECK_TYPE(ArmPlugin::opset::Elu, op);
    CHECK_TYPE(ArmPlugin::opset::ArmTranspose, op);
    CHECK_TYPE(ArmPlugin::opset::Softmax, op);
    CHECK_TYPE(ArmPlugin::opset::ArmSplit, op);
    CHECK_TYPE(ArmPlugin::opset::LRN, op);
    CHECK_TYPE(ArmPlugin::opset::Minimum, op);
    CHECK_TYPE(ArmPlugin::opset::Maximum, op);
    CHECK_TYPE(ArmPlugin::opset::ArmStridedSlice, op);
    CHECK_TYPE(ArmPlugin::opset::Negative, op);
    CHECK_TYPE(ArmPlugin::opset::Floor, op);
    CHECK_TYPE(ArmPlugin::opset::Exp, op);
    CHECK_TYPE(ArmPlugin::opset::Divide, op);
    CHECK_TYPE(ArmPlugin::opset::MatMul, op);
    CHECK_TYPE(ArmPlugin::opset::ArmMatMulBias, op);
    CHECK_TYPE(ArmPlugin::opset::Pad, op);
    CHECK_TYPE(ArmPlugin::opset::BatchNormInference, op);
    CHECK_TYPE(ArmPlugin::opset::HSwish, op);
    CHECK_TYPE(ArmPlugin::opset::SoftPlus, op);
    CHECK_TYPE(ArmPlugin::opset::Log, op);
    CHECK_TYPE(ArmPlugin::opset::Sin, op);
    CHECK_TYPE(ArmPlugin::opset::ShuffleChannels, op);
    CHECK_TYPE(ArmPlugin::opset::Power, op);
    CHECK_TYPE(ArmPlugin::opset::SquaredDifference, op);
    CHECK_TYPE(ArmPlugin::opset::ReduceMean, op);
    CHECK_TYPE(ArmPlugin::opset::ReduceSum, op);
//    Failed native layer in ACL for FP16 precision 
//    CHECK_TYPE(ArmPlugin::opset::ReduceProd, op);
    CHECK_TYPE(ArmPlugin::opset::ReduceMin, op);
    CHECK_TYPE(ArmPlugin::opset::ReduceMax, op);
    CHECK_TYPE(ArmPlugin::opset::ArmInterpolate, op);
    CHECK_TYPE(ArmPlugin::opset::ArmMVN, op);
    CHECK_TYPE(ArmPlugin::opset::ArmNormalizeL2, op);
    CHECK_TYPE(ArmPlugin::opset::DepthToSpace, op);
    CHECK_TYPE(ArmPlugin::opset::SpaceToDepth, op);
    CHECK_TYPE(ArmPlugin::opset::Equal, op);
    CHECK_TYPE(ArmPlugin::opset::NotEqual, op);
    CHECK_TYPE(ArmPlugin::opset::Less, op);
    CHECK_TYPE(ArmPlugin::opset::LessEqual, op);
    CHECK_TYPE(ArmPlugin::opset::Greater, op);
    CHECK_TYPE(ArmPlugin::opset::GreaterEqual, op);
    CHECK_TYPE(ArmPlugin::opset::Select, op);
    CHECK_TYPE(ArmPlugin::opset::ReorgYolo, op);
    CHECK_TYPE(ArmPlugin::opset::BatchToSpace, op);
    CHECK_TYPE(ArmPlugin::opset::SpaceToBatch, op);
    CHECK_TYPE(ArmPlugin::opset::ArmConvert, op);
    CHECK_TYPE(ArmPlugin::opset::ArmConcat, op);
    CHECK_TYPE(ArmPlugin::opset::ArmGather, op);
    CHECK_TYPE(ArmPlugin::opset::ArmFFT, op);
    CHECK_TYPE(ArmPlugin::opset::ArmQuantize, op);
    CHECK_TYPE(ArmPlugin::opset::ArmDequantize, op);
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
