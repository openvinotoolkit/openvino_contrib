// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace nvidia_gpu {

class TensorMappingContext {
    using TensorVec = std::vector<std::shared_ptr<ov::Tensor>>;
    using MappingMap = std::map<std::string, std::size_t>;

public:
    TensorMappingContext(const TensorVec& inputs,
                         const MappingMap& inputMapping,
                         const TensorVec& outputs,
                         const MappingMap& outputMapping)
        : blob_inputs{inputs}, inputs_mapping{inputMapping}, blob_outputs{outputs}, outputs_mapping{outputMapping} {}
    /**
     * @brief get_input_tensor(name) returns an tensor blob with the given name
     */
    inline std::shared_ptr<ov::Tensor> get_input_tensor(const std::string& input_name) const {
        return blob_inputs.at(inputs_mapping.at(input_name));
    }
    /**
     * @brief get_output_tensor(name) returns an output tensor with the given name
     */
    inline std::shared_ptr<ov::Tensor> get_output_tensor(const std::string& output_name) const {
        return blob_outputs.at(outputs_mapping.at(output_name));
    }
    /**
     * @brief has_input_tensor(name) returns true if it contains an input tensor with the given name
     */
    inline bool has_input_tensor(const std::string& input_name) const noexcept {
        return inputs_mapping.find(input_name) != inputs_mapping.end();
    }
    /**
     * @brief has_output_tensor(name) returns true if contains an output tensor with the given name
     */
    inline bool has_output_tensor(const std::string& output_name) const noexcept {
        return outputs_mapping.find(output_name) != outputs_mapping.end();
    }

private:
    const TensorVec& blob_inputs;
    const MappingMap& inputs_mapping;
    const TensorVec& blob_outputs;
    const MappingMap& outputs_mapping;
};

}  // namespace nvidia_gpu
}  // namespace ov
