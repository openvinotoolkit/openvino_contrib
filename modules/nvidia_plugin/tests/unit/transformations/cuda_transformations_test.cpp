// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/runtime.hpp>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "transformer/cuda_graph_transformer.hpp"

using namespace testing;

TEST(TransformationTests, cuda_transformations_f16) {
    std::shared_ptr<ov::Model> model, model_ref;
    {
        // Example model
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto divide_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2});
        auto divide = std::make_shared<ov::op::v1::Divide>(data, divide_constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{divide}, ov::ParameterVector{data});

        // Run transformation
        const CUDA::Device device{};
        const auto config = ov::nvidia_gpu::Configuration(ov::AnyMap{ov::hint::inference_precision(ov::element::f16)});
        ov::nvidia_gpu::GraphTransformer().transform(device, model, config);

        // Check that after applying transformation all runtime info attributes was correctly propagated
        ASSERT_NO_THROW(check_rt_info(model));

        if (!CUDA::isHalfSupported(device)) {
            GTEST_SKIP() << "f16 precision isn't fully supported on the device";
        }
    }

    {
        // Example reference model
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});
        auto convert_f16 = std::make_shared<ov::op::v0::Convert>(data, ov::element::f16);
        auto mul_constant = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{1, 1, 1}, {0.5});
        auto mul = std::make_shared<ov::op::v1::Multiply>(convert_f16, mul_constant);
        auto convert_f32 = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f32);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{convert_f32}, ov::ParameterVector{data});
    }
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}
