// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cancellation_token.hpp>

using namespace ov::nvidia_gpu;

class CancellationTokenTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CancellationTokenTest, Cancel_No_Throw) {
    CancellationToken token{};
    ASSERT_NO_THROW(token.cancel());
}

TEST_F(CancellationTokenTest, Cancel_No_Throw_Callback) {
    bool is_cancelled = false;
    CancellationToken token{[&is_cancelled] { is_cancelled = true; }};
    ASSERT_NO_THROW(token.cancel());
    ASSERT_TRUE(is_cancelled);
}