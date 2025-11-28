// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transforms/pipeline.hpp"

namespace ov {
namespace metal_plugin {
namespace transforms {

std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model) {
    // No transformations are applied yet; return the original model.
    return model;
}

}  // namespace transforms
}  // namespace metal_plugin
}  // namespace ov
