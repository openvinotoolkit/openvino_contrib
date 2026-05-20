// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <regex>
#include <vector>

const std::vector<std::regex>& disabled_test_patterns() {
    static const std::vector<std::regex> patterns{
        // GFX tests must compile either explicit GFX target or TEMPLATE reference.
        // OpenVINO's default-device path can select a host plugin and violates the
        // GFX no-CPU-inference contract.
        std::regex(R"(.*OVCompiledModelBaseTest\.canCompileModelToDefaultDevice.*)"),
        // HETERO synthetic splits not yet supported by GFX backend
        std::regex(R"(.*OVHeteroSyntheticTest.*)")};
    return patterns;
}
