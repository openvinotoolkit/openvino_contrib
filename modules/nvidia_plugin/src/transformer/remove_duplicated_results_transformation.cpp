// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "remove_duplicated_results_transformation.hpp"

#include <cuda_op_buffers_extractor.hpp>
#include <exec_graph_info.hpp>
#include <gsl/span_ext>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <openvino/op/matmul.hpp>
#include <openvino/op/transpose.hpp>

namespace ov::nvidia_gpu::pass {
bool RemoveDuplicatedResultsTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(RemoveDuplicatedResultsTransformation);
    std::unordered_set<std::shared_ptr<ov::op::v0::Result>> duplicated_results;
    for (auto& result : f->get_results()) {
        if (duplicated_results.count(result) > 0) {
            continue;
        }
        auto result_input = result->input(0);
        auto source_output = result_input.get_source_output();
        for (const auto& in : source_output.get_target_inputs()) {
            if (auto other_result = dynamic_cast<ov::op::v0::Result*>(in.get_node())) {
                if (result.get() != other_result) {
                    duplicated_results.insert(
                        std::dynamic_pointer_cast<ov::op::v0::Result>(other_result->shared_from_this()));
                }
            }
        }
    }

    if (!duplicated_results.empty()) {
        for (const auto& res : duplicated_results) {
            f->remove_result(res);
        }
        return true;
    }

    return false;
}

}  // namespace ov::nvidia_gpu::pass
