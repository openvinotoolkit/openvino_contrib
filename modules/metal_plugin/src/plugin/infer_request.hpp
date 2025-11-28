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
    std::vector<ov::ProfilingInfo> get_profiling_info() const override { return {}; }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override { return {}; }

private:
    const std::shared_ptr<const CompiledModel> get_compiled_model_typed() const;
};

}  // namespace metal_plugin
}  // namespace ov
