// clang-format off
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_tests_instances/test_utils.hpp"
#include "integration/test_constants.hpp"

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

using GfxCompileModelCacheTestBase = ov::test::utils::GfxSkippedTests<CompileModelCacheTestBase>;
using GfxCompileModelLoadFromCacheTest = ov::test::utils::GfxSkippedTests<CompileModelLoadFromCacheTest>;
using GfxCompileModelLoadFromMemoryTestBase =
    ov::test::utils::GfxSkippedTests<CompileModelLoadFromMemoryTestBase>;
using GfxCompileModelWithCacheEncryptionTest =
    ov::test::utils::GfxSkippedTests<CompileModelWithCacheEncryptionTest>;
using GfxOVBasicPropertiesTestsP = ov::test::utils::GfxSkippedTests<OVBasicPropertiesTestsP>;
using GfxOVCheckGetSupportedROMetricsPropsTests =
    ov::test::utils::GfxSkippedTests<OVCheckGetSupportedROMetricsPropsTests>;
using GfxOVClassModelOptionalTestP = ov::test::utils::GfxSkippedTests<OVClassModelOptionalTestP>;
using GfxOVClassModelTestP = ov::test::utils::GfxSkippedTests<OVClassModelTestP>;
using GfxOVClassQueryModelTest = ov::test::utils::GfxSkippedTests<OVClassQueryModelTest>;
using GfxOVGetMetricPropsTest = ov::test::utils::GfxSkippedTests<OVGetMetricPropsTest>;
using GfxOVHeteroSyntheticTest = ov::test::utils::GfxSkippedTests<OVHeteroSyntheticTest>;
using GfxOVPropertiesDefaultSupportedTests =
    ov::test::utils::GfxSkippedTests<OVPropertiesDefaultSupportedTests>;
using GfxOVPropertiesDefaultTests = ov::test::utils::GfxSkippedTests<OVPropertiesDefaultTests>;
using GfxOVPropertiesIncorrectTests = ov::test::utils::GfxSkippedTests<OVPropertiesIncorrectTests>;
using GfxOVPropertiesTests = ov::test::utils::GfxSkippedTests<OVPropertiesTests>;
using GfxOVRemoteTest = ov::test::utils::GfxSkippedTests<OVRemoteTest>;
using GfxOVVersionTest = ov::test::utils::GfxSkippedTests<VersionTests>;

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
                         GfxOVPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(inproperties)),
                         OVPropertiesIncorrectTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVPropertiesDefaultTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(default_properties)),
                         OVPropertiesDefaultTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVPropertiesDefaultSupportedTests,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(properties)),
                         OVPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVGetMetricPropsTest,
                         GfxOVGetMetricPropsTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(
    smoke_OVCheckGetSupportedROMetricsPropsTests,
    GfxOVCheckGetSupportedROMetricsPropsTests,
    ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_GFX),
                       ::testing::ValuesIn(OVCheckGetSupportedROMetricsPropsTests::configureProperties(
                           {ov::device::full_name.name()}))),
    OVCheckGetSupportedROMetricsPropsTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVBasicPropertiesTestsP,
                         GfxOVBasicPropertiesTestsP,
                         ::testing::Values(std::make_pair("openvino_gfx_plugin",
                                                          ov::test::utils::DEVICE_GFX)));

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVVersionTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX),
                         VersionTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxCompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(std::size_t{1}),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         GfxCompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::ValuesIn(test_targets), ::testing::ValuesIn(cache_configs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         GfxCompileModelLoadFromCacheTest,
                         ::testing::Combine(::testing::ValuesIn(test_targets), ::testing::ValuesIn(cache_configs)),
                         CompileModelLoadFromCacheTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Template,
                         GfxCompileModelWithCacheEncryptionTest,
                         testing::ValuesIn(test_targets),
                         CompileModelWithCacheEncryptionTest::getTestCaseName);

// Core integration / query model
INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelTestP,
                         GfxOVClassModelTestP,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassModelOptionalTestP,
                         GfxOVClassModelOptionalTestP,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

INSTANTIATE_TEST_SUITE_P(smoke_OVClassQueryModelTest,
                         GfxOVClassQueryModelTest,
                         ::testing::Values(ov::test::utils::DEVICE_GFX));

// Remote tensors
auto metal_remote_configs = []() {
    return std::vector<ov::AnyMap>{{}};
};

std::vector<std::pair<ov::AnyMap, ov::AnyMap>> generate_remote_params() {
    return std::vector<std::pair<ov::AnyMap, ov::AnyMap>>{{{}, {}}};
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVRemoteTest,
                         ::testing::Combine(::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GFX),
                                            ::testing::ValuesIn(metal_remote_configs()),
                                            ::testing::ValuesIn(generate_remote_params())),
                         OVRemoteTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         GfxOVHeteroSyntheticTest,
                         ::testing::Combine(::testing::Values(std::vector<ov::test::behavior::PluginParameter>{
                                                {"GFX", "openvino_gfx_plugin"},
                                                {"TEMPLATE", "openvino_template_plugin"}}),
                                            ::testing::ValuesIn(OVHeteroSyntheticTest::_singleMajorNodeFunctions)),
                         OVHeteroSyntheticTest::getTestCaseName);

}  // namespace
