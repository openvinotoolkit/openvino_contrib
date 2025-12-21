// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "metal_test_utils.hpp"
#include "test_constants.hpp"

#include <string>
#include <utility>
#include <vector>

#include "behavior/ov_plugin/caching_tests.hpp"
#include "behavior/ov_plugin/core_integration_sw.hpp"
#include "behavior/ov_plugin/hetero_synthetic.hpp"
#include "behavior/ov_plugin/life_time.hpp"
#include "behavior/ov_plugin/properties_tests.hpp"
#include "behavior/ov_plugin/query_model.hpp"
#include "behavior/ov_plugin/remote.hpp"
#include "behavior/ov_plugin/version.hpp"

using ov::test::behavior::CompileModelCacheTestBase;
using ov::test::behavior::CompileModelLoadFromCacheTest;
using ov::test::behavior::CompileModelLoadFromMemoryTestBase;
using ov::test::behavior::CompileModelWithCacheEncryptionTest;
using ov::test::behavior::OVBasicPropertiesTestsP;
using ov::test::behavior::OVCheckGetSupportedROMetricsPropsTests;
using ov::test::behavior::OVClassModelOptionalTestP;
using ov::test::behavior::OVClassModelTestP;
using ov::test::behavior::OVClassQueryModelTest;
using ov::test::behavior::OVGetMetricPropsTest;
using ov::test::behavior::OVHeteroSyntheticTest;
using ov::test::behavior::OVPropertiesDefaultSupportedTests;
using ov::test::behavior::OVPropertiesDefaultTests;
using ov::test::behavior::OVPropertiesIncorrectTests;
using ov::test::behavior::OVPropertiesTests;
using ov::test::OVRemoteTest;
using ov::test::behavior::VersionTests;

using MetalCompileModelCacheTestBase = ov::test::utils::GfxSkippedTests<CompileModelCacheTestBase>;
using MetalCompileModelLoadFromCacheTest = ov::test::utils::GfxSkippedTests<CompileModelLoadFromCacheTest>;
using MetalCompileModelLoadFromMemoryTestBase =
    ov::test::utils::GfxSkippedTests<CompileModelLoadFromMemoryTestBase>;
using MetalCompileModelWithCacheEncryptionTest =
    ov::test::utils::GfxSkippedTests<CompileModelWithCacheEncryptionTest>;
using MetalOVBasicPropertiesTestsP = ov::test::utils::GfxSkippedTests<OVBasicPropertiesTestsP>;
using MetalOVCheckGetSupportedROMetricsPropsTests =
    ov::test::utils::GfxSkippedTests<OVCheckGetSupportedROMetricsPropsTests>;
using MetalOVClassModelOptionalTestP = ov::test::utils::GfxSkippedTests<OVClassModelOptionalTestP>;
using MetalOVClassModelTestP = ov::test::utils::GfxSkippedTests<OVClassModelTestP>;
using MetalOVClassQueryModelTest = ov::test::utils::GfxSkippedTests<OVClassQueryModelTest>;
using MetalOVGetMetricPropsTest = ov::test::utils::GfxSkippedTests<OVGetMetricPropsTest>;
using MetalOVHeteroSyntheticTest = ov::test::utils::GfxSkippedTests<OVHeteroSyntheticTest>;
using MetalOVPropertiesDefaultSupportedTests =
    ov::test::utils::GfxSkippedTests<OVPropertiesDefaultSupportedTests>;
using MetalOVPropertiesDefaultTests = ov::test::utils::GfxSkippedTests<OVPropertiesDefaultTests>;
using MetalOVPropertiesIncorrectTests = ov::test::utils::GfxSkippedTests<OVPropertiesIncorrectTests>;
using MetalOVPropertiesTests = ov::test::utils::GfxSkippedTests<OVPropertiesTests>;
using MetalOVRemoteTest = ov::test::utils::GfxSkippedTests<OVRemoteTest>;
using MetalOVVersionTest = ov::test::utils::GfxSkippedTests<VersionTests>;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::device::id("UNSUPPORTED_DEVICE_ID_STRING")},
};

const std::vector<ov::AnyMap> default_properties = {
    {ov::enable_profiling(false)},
    {ov::device::id(0)},
};

const std::vector<ov::AnyMap> properties = {
    {ov::enable_profiling(true)},
    {ov::device::id(0)},
};

const std::vector<ov::AnyMap> cache_configs = {
    {ov::num_streams(2)},
};

const std::vector<std::string> test_targets = {ov::test::utils::DEVICE_GFX};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest,
                         MetalOVGetMetricPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckGetSupportedROMetricsPropsTests,
    MetalOVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVBasicPropertiesTestsP,
                         MetalOVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_gfx_plugin",
                                                          ov::test::utils::DEVICE_GFX)));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVVersionTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX),
                         VersionTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalCompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(std::size_t{1}),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         MetalCompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::ValuesIn(test_targets), ::testing::ValuesIn(cache_configs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         MetalCompileModelLoadFromCacheTest,
                         ::testing::Combine(::testing::ValuesIn(test_targets), ::testing::ValuesIn(cache_configs)),
                         CompileModelLoadFromCacheTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         MetalCompileModelWithCacheEncryptionTest,
                         testing::ValuesIn(test_targets),
                         CompileModelWithCacheEncryptionTest::getTestCaseName);

// Core integration / query model
INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelTestP,
                         MetalOVClassModelTestP,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelOptionalTestP,
                         MetalOVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest,
                         MetalOVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

// Remote tensors
auto metal_remote_configs = []() {
    return std::vector<ov::AnyMap>{{}};
};

std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params() {
    return std::vector<std::pair<ov::AnyMap, ov::AnyMap>>{{{}, {}}};
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVRemoteTest,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(metal_remote_configs()),
                                            ::testing::ValuesIn(generate_remote_params())),
                         OVRemoteTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         MetalOVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<ov::test::behavior::PluginParameter>{
                                                {"GFX", "openvino_gfx_plugin"},
                                                {"CPU", "openvino_intel_cpu_plugin"}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::_singleMajorNodeFunctions)),
                         OVHeteroSyntheticTest::getTestCaseName);

}  // namespace
