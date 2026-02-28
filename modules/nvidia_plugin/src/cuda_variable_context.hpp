// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_variable_state.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Maps variable IDs to CudaVariableState instances.
 * Owned by CudaInferRequest. Passed by reference to InferenceRequestContext.
 */
class CudaVariableContext {
public:
    void register_variable(const std::string& variable_id, CudaVariableState::Ptr state) {
        states_.emplace(variable_id, std::move(state));
    }

    CudaVariableState::Ptr get_variable_state(const std::string& variable_id) const {
        auto it = states_.find(variable_id);
        OPENVINO_ASSERT(it != states_.end(), "Variable state not found: ", variable_id);
        return it->second;
    }

    bool has_variable_state(const std::string& variable_id) const {
        return states_.count(variable_id) > 0;
    }

    std::vector<ov::SoPtr<ov::IVariableState>> query_states() const {
        std::vector<ov::SoPtr<ov::IVariableState>> result;
        result.reserve(states_.size());
        for (const auto& [id, state] : states_) {
            result.emplace_back(state, nullptr);
        }
        return result;
    }

    bool empty() const { return states_.empty(); }

private:
    std::unordered_map<std::string, CudaVariableState::Ptr> states_;
};

}  // namespace nvidia_gpu
}  // namespace ov
