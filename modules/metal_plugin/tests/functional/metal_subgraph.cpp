// clang-format off
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metal_test_utils.hpp"
#include "test_constants.hpp"

#include "shared_test_classes/subgraph/stateful_model.hpp"

using MetalStatefulModelStateInLoopBody =
    ov::test::utils::MetalSkippedTests<ov::test::StatefulModelStateInLoopBody>;

INSTANTIATE_TEST_SUITE_P(smoke,
                         MetalStatefulModelStateInLoopBody,
                         ::testing::Values(ov::test::utils::DEVICE_METAL));
