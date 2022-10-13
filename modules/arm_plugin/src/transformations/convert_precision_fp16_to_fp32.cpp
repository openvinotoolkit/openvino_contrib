// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/common_optimizations/convert_precision_fp16_to_fp32.hpp"

#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset8.hpp>

bool ArmPlugin::pass::ConvertPrecisionFP16ToFP32::run_on_model(const std::shared_ptr<ov::Model>& m) {
    const auto ordered_ops = m->get_ordered_ops();
    for (const auto& op : ordered_ops) {
        bool is_internal_arm_op = strcmp(op->get_type_info().version_id, "arm_opset") == 0;
        bool is_result_op = std::dynamic_pointer_cast<opset8::Result>(op) != nullptr;
        if (is_internal_arm_op || is_result_op) {
            continue;
        }

        bool convert_for_outputs_required = false;
        for (const auto& input : op->inputs()) {
            if (input.get_element_type() == ov::element::f16) {
                auto parent_output = input.get_source_output();
                auto parent_convert = std::dynamic_pointer_cast<opset8::Convert>(parent_output.get_node_shared_ptr());
                if (parent_convert && parent_convert->get_rt_info().count("fp16_to_fp32_transformation") != 0) {
                    input.replace_source_output(parent_convert->input_value(0));
                } else {
                    auto convert = std::make_shared<opset8::Convert>(input.get_source_output(), ov::element::f32);
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
                    auto convert = std::make_shared<opset8::Convert>(output, ov::element::f16);

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
