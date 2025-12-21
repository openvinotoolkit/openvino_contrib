// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metal_test_utils.hpp"
#include "test_constants.hpp"

#include <tuple>
#include <vector>

#include "behavior/compiled_model/compiled_model_base.hpp"
#include "behavior/compiled_model/import_export.hpp"
#include "behavior/compiled_model/properties.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"

using ov::test::behavior::OVClassCompiledModelGetConfigTest;
using ov::test::behavior::OVClassCompiledModelGetIncorrectPropertyTest;
using ov::test::behavior::OVClassCompiledModelGetPropertyTest;
using ov::test::behavior::OVClassCompiledModelGetPropertyTest_EXEC_DEVICES;
using ov::test::behavior::OVClassCompiledModelSetIncorrectConfigTest;
using ov::test::behavior::OVCompiledGraphImportExportTest;
using ov::test::behavior::OVCompiledModelBaseTest;
using ov::test::behavior::OVCompiledModelBaseTestOptional;
using ov::test::behavior::OVCompiledModelIncorrectDevice;
using ov::test::behavior::OVCompiledModelPropertiesDefaultSupportedTests;
using ov::test::behavior::OVClassCompiledModelPropertiesTests;
using ov::test::behavior::OVClassCompiledModelPropertiesDefaultTests;
using ov::test::behavior::OVClassCompiledModelPropertiesIncorrectTests;
using ov::test::behavior::OVClassCompiledModelEmptyPropertiesTests;

using MetalOVCompiledModelBaseTest = ov::test::utils::GfxSkippedTests<OVCompiledModelBaseTest>;
using MetalOVCompiledModelBaseTestOptional = ov::test::utils::GfxSkippedTests<OVCompiledModelBaseTestOptional>;
using MetalOVCompiledModelIncorrectDevice = ov::test::utils::GfxSkippedTests<OVCompiledModelIncorrectDevice>;
using MetalOVCompiledModelPropertiesDefaultSupportedTests =
    ov::test::utils::GfxSkippedTests<OVCompiledModelPropertiesDefaultSupportedTests>;
using MetalOVCompiledModelPropertiesTests = ov::test::utils::GfxSkippedTests<OVClassCompiledModelPropertiesTests>;
using MetalOVClassCompiledModelPropertiesDefaultTests =
    ov::test::utils::GfxSkippedTests<OVClassCompiledModelPropertiesDefaultTests>;
using MetalOVClassCompiledModelPropertiesIncorrectTests =
    ov::test::utils::GfxSkippedTests<OVClassCompiledModelPropertiesIncorrectTests>;
using MetalOVClassCompiledModelEmptyPropertiesTests =
    ov::test::utils::GfxSkippedTests<OVClassCompiledModelEmptyPropertiesTests>;
using MetalOVClassCompiledModelGetPropertyTest = ov::test::utils::GfxSkippedTests<OVClassCompiledModelGetPropertyTest>;
using MetalOVClassCompiledModelGetIncorrectPropertyTest =
    ov::test::utils::GfxSkippedTests<OVClassCompiledModelGetIncorrectPropertyTest>;
using MetalOVClassCompiledModelGetConfigTest = ov::test::utils::GfxSkippedTests<OVClassCompiledModelGetConfigTest>;
using MetalOVClassCompiledModelSetIncorrectConfigTest =
    ov::test::utils::GfxSkippedTests<OVClassCompiledModelSetIncorrectConfigTest>;
using MetalOVCompiledGraphImportExportTest = ov::test::utils::GfxSkippedTests<OVCompiledGraphImportExportTest>;
using MetalOVClassCompiledModelGetPropertyTest_EXEC_DEVICES =
    ov::test::utils::GfxSkippedTests<OVClassCompiledModelGetPropertyTest_EXEC_DEVICES>;

namespace {

const std::vector<ov::AnyMap> configs = {
    {},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVCompiledModelBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledModelBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVCompiledModelBaseTestOptional,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledModelBaseTestOptional::getTestCaseName);

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> default_properties = {
    {ov::enable_profiling(false)},
    {{ov::loaded_from_cache.name(), false}},
    {ov::device::id("0")},
};

const std::vector<ov::AnyMap> properties = {
    {ov::enable_profiling(true)},
    {ov::device::id("0")},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVClassCompiledModelPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(default_properties)),
                         OVClassCompiledModelPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVCompiledModelPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_GFX),
                         OVCompiledModelPropertiesDefaultSupportedTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelEmptyPropertiesTests,
                         MetalOVClassCompiledModelEmptyPropertiesTests,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice,
                         MetalOVCompiledModelIncorrectDevice,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

std::vector<std::string> devices = {ov::test::utils::DEVICE_GFX};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         MetalOVClassCompiledModelGetPropertyTest,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetIncorrectPropertyTest,
                         MetalOVClassCompiledModelGetIncorrectPropertyTest,
                         ::testing::ValuesIn(devices));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetConfigTest,
                         MetalOVClassCompiledModelGetConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelSetIncorrectConfigTest,
                         MetalOVClassCompiledModelSetIncorrectConfigTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::i8,  ov::element::i16, ov::element::i32, ov::element::i64, ov::element::u8,
    ov::element::u16, ov::element::u32, ov::element::u64, ov::element::f16, ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(configs)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

const std::vector<std::tuple<std::string, std::pair<ov::AnyMap, std::string>>>
    GetMetricTest_ExecutionDevice_TEMPLATE = {{ov::test::utils::DEVICE_GFX, std::make_pair(ov::AnyMap{}, "TEMPLATE.0")}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         MetalOVClassCompiledModelGetPropertyTest_EXEC_DEVICES,
                         ::testing::ValuesIn(GetMetricTest_ExecutionDevice_TEMPLATE));

}  // namespace
