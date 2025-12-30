// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <istream>
#include <memory>
#include <ostream>

#include "openvino/core/model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
class ICore;
namespace gfx_plugin {

void write_model_to_stream(const std::shared_ptr<const ov::Model>& model, std::ostream& stream);

std::shared_ptr<ov::Model> read_model_from_stream(const std::shared_ptr<ov::ICore>& core, std::istream& stream);

std::shared_ptr<ov::Model> read_model_from_buffer(const std::shared_ptr<ov::ICore>& core, const ov::Tensor& model);

}  // namespace gfx_plugin
}  // namespace ov
