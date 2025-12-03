// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "runtime/backend.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace metal_plugin {

struct Segment;

// Experimental MLIR-themed backend. Currently does a minimal MatMul path using generated MSL.
class MlirBackend final : public MetalBackend {
public:
    explicit MlirBackend(const std::shared_ptr<const ov::Model>& model,
                         const std::shared_ptr<const ov::Model>& original_model,
                         ov::element::Type inference_precision);
    ~MlirBackend();
    void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) override;
    bool has_segment() const;
    bool segment_io_is_model_io() const;
    const Segment& get_segment() const;
    std::vector<ov::Tensor> run_segment(const Segment& seg, const std::vector<ov::Tensor>& inputs);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace metal_plugin
}  // namespace ov
