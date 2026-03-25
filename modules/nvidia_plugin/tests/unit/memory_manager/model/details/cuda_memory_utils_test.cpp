// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/details/cuda_memory_utils.hpp"

#include <gtest/gtest.h>

TEST(MemoryUtils, ApplyAllignment) {
    const size_t allignment = 256;

    using namespace ov::nvidia_gpu;
    ASSERT_EQ(applyAllignment(0), 0);
    ASSERT_EQ(applyAllignment(1), allignment);

    ASSERT_EQ(applyAllignment(1 * allignment - 1), 1 * allignment);
    ASSERT_EQ(applyAllignment(1 * allignment), 1 * allignment);
    ASSERT_EQ(applyAllignment(1 * allignment + 1), 2 * allignment);

    ASSERT_EQ(applyAllignment(2 * allignment - 1), 2 * allignment);
    ASSERT_EQ(applyAllignment(2 * allignment), 2 * allignment);
    ASSERT_EQ(applyAllignment(2 * allignment + 1), 3 * allignment);
}
