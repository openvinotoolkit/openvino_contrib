// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cancellation_token.hpp>

using namespace CUDAPlugin;

class CancellationTokenTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CancellationTokenTest, Cancel_Throw) {
    CancellationToken token{};
    token.Cancel();
    ASSERT_THROW(token.Check(), InferenceEngine::details::InferenceEngineException);
}

TEST_F(CancellationTokenTest, Cancel_Throw_Callback) {
    bool is_cancelled = false;
    CancellationToken token{[&is_cancelled] { is_cancelled = true; }};
    token.Cancel();
    ASSERT_THROW(token.Check(), InferenceEngine::details::InferenceEngineException);
    ASSERT_TRUE(is_cancelled);
}