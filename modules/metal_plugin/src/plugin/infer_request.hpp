// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace metal_plugin {

class CompiledModel;

class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model);

    void infer() override;
    void set_input_tensor(const ov::Tensor& tensor);
    void set_input_tensor(size_t idx, const ov::Tensor& tensor);
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override { return {}; }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override { return {}; }

private:
    const std::shared_ptr<const CompiledModel> get_compiled_model_typed() const;

    std::vector<ov::Tensor> m_bound_inputs;
};

}  // namespace metal_plugin
}  // namespace ov
