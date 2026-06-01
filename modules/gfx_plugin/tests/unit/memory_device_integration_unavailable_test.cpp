// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "plugin/gfx_backend_config.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

TEST(GfxRunDeviceIntegration, NoHostRoundTripUntilOutputRequested) {
    EXPECT_FALSE(kGfxBackendMetalAvailable)
        << "The native Metal memory integration adapter must not be linked "
           "when the Metal backend is present.";
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
