// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/manager.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {

class GfxLayoutCleanup : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("GfxLayoutCleanup");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
