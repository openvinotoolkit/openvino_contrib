// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels/details/typed_functor.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace ov::nvidia_gpu::kernel;

using TFuncPtr = void (*)();

template <typename T1, typename T2>
struct Functor {
    static void function() {}
};

struct TypedFunctorTest : public ::testing::Test {
    static constexpr TypedFunctor<Functor, TFuncPtr, DIM_2D> combinations{};
};

TEST_F(TypedFunctorTest, outOfRange) {
    EXPECT_THROW(
        {
            try {
                [[maybe_unused]] auto func_ptr = combinations[Type_t::u64][static_cast<Type_t>(100000)];
            } catch (const std::exception& e) {
                auto msg = e.what();
                EXPECT_THAT(
                    msg,
                    ::testing::HasSubstr("TypedFunctor[Dimension=1]: Type = 100000 is not supported by TypedFunctor"));
                throw;
            }
        },
        std::exception);
}
