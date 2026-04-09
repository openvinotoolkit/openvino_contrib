// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <common_test_utils/test_constants.hpp>
#include <cuda_test_constants.hpp>

#include "behavior/compiled_model/import_export.hpp"

using namespace ov::test::behavior;
namespace {
const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::f16, ov::element::f32, ov::element::i8,
    // TODO: Add additional network precisions
};
const std::vector<ov::AnyMap> configs = {
    {},
};
const std::vector<ov::AnyMap> multiConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NVIDIA)}};

const std::vector<ov::AnyMap> heteroConfigs = {{ov::device::priorities(ov::test::utils::DEVICE_NVIDIA)}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_NVIDIA),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(multiConfigs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Hetero_BehaviorTests,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::ValuesIn(heteroConfigs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

}  // namespace