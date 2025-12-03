// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

#include "openvino/core/core_visibility.hpp"

std::vector<std::string> disabledTestPatterns() {
    std::vector<std::string> retVector{
        // HETERO synthetic splits not yet supported by METAL backend
        R"(.*OVHeteroSyntheticTest.*)"};
    return retVector;
}
