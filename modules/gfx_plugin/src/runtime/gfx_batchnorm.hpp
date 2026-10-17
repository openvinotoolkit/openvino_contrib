// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <vector>

namespace ov {
namespace gfx_plugin {

struct BatchNormParams {
    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> mean;
    std::vector<float> var;
    float epsilon = 0.0f;

    bool empty() const {
        return gamma.empty() || beta.empty() || mean.empty() || var.empty();
    }
};

}  // namespace gfx_plugin
}  // namespace ov
