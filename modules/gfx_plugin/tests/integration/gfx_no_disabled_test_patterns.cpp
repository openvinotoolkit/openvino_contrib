// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include <vector>

const std::vector<std::regex>& disabled_test_patterns() {
    // OpenVINO shared tests require this hook; GFX production tests keep it empty.
    static const std::vector<std::regex> patterns{};
    return patterns;
}
