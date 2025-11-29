// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/backend.hpp"

namespace ov {
namespace metal_plugin {

// Experimental MLIR-themed backend. Currently does a minimal MatMul path using generated MSL.
class MlirBackend final : public MetalBackend {
public:
    explicit MlirBackend(const std::shared_ptr<const ov::Model>& model);
    ~MlirBackend();
    void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) override;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace metal_plugin
}  // namespace ov
