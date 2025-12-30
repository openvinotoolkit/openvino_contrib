// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "plugin/gfx_backend_config.hpp"

TEST(VulkanBackendTest, CompileToSpirv) {
    EXPECT_FALSE(ov::gfx_plugin::kGfxBackendVulkanAvailable);
}
