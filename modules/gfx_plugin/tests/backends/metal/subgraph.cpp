// clang-format off
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_tests_instances/test_utils.hpp"
#include "integration/test_constants.hpp"

#include "shared_test_classes/subgraph/stateful_model.hpp"

namespace ov::test {

INSTANTIATE_TEST_SUITE_P(smoke, StaticShapeStatefulModel, ::testing::Values(ov::test::utils::DEVICE_GFX));
INSTANTIATE_TEST_SUITE_P(smoke, StaticShapeTwoStatesModel, ::testing::Values(ov::test::utils::DEVICE_GFX));
INSTANTIATE_TEST_SUITE_P(smoke, DynamicShapeStatefulModelDefault, ::testing::Values(ov::test::utils::DEVICE_GFX));
INSTANTIATE_TEST_SUITE_P(smoke, DynamicShapeStatefulModelParam, ::testing::Values(ov::test::utils::DEVICE_GFX));
INSTANTIATE_TEST_SUITE_P(smoke,
                         DynamicShapeStatefulModelStateAsInp,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));
INSTANTIATE_TEST_SUITE_P(smoke,
                         StatefulModelStateInLoopBody,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

}  // namespace ov::test
