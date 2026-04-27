// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/openvino.hpp"

#include "transforms/pipeline.hpp"

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"

namespace {

ov::Tensor infer_with_template(const std::shared_ptr<const ov::Model>& model) {
    ov::Core core;
    ov::test::utils::register_template_plugin(core);
    auto compiled = core.compile_model(model, "TEMPLATE");
    auto request = compiled.create_infer_request();
    request.infer();
    return request.get_output_tensor(0);
}

}  // namespace

TEST(GfxTransforms, MergeZeroPadIntoConvolution) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 3, 8, 8});
    auto pads_begin = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 1, 2});
    auto pads_end = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 3, 4});
    auto pad_value = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0f});
    auto pad = std::make_shared<ov::op::v1::Pad>(input, pads_begin, pads_end, pad_value, ov::op::PadMode::CONSTANT);

    std::vector<float> weights_data(8 * 3 * 3 * 3, 0.1f);
    auto weights = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{8, 3, 3, 3}, weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(pad,
                                                           weights,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{0, 0},
                                                           ov::CoordinateDiff{0, 0},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT);
    auto result = std::make_shared<ov::op::v0::Result>(conv);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "pad_conv_merge");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    bool has_pad = false;
    std::shared_ptr<ov::op::v1::Convolution> merged_conv;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Pad>(node)) {
            has_pad = true;
        }
        if (auto c = ov::as_type_ptr<ov::op::v1::Convolution>(node)) {
            merged_conv = c;
        }
    }

    ASSERT_FALSE(has_pad);
    ASSERT_TRUE(merged_conv);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(merged_conv->input_value(0).get_node_shared_ptr()));
    EXPECT_EQ(merged_conv->get_input_partial_shape(0), ov::PartialShape({1, 3, 8, 8}));
    EXPECT_EQ(merged_conv->get_pads_begin(), ov::CoordinateDiff({1, 2}));
    EXPECT_EQ(merged_conv->get_pads_end(), ov::CoordinateDiff({3, 4}));
}

TEST(GfxTransforms, FoldTransposeSoftmaxInverseTranspose) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16, 8400});
    auto perm0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::op::v1::Transpose>(input, perm0);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(transpose0, 3);
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(softmax, perm1);
    auto result = std::make_shared<ov::op::v0::Result>(transpose1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "softmax_transpose_fold");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    int transpose_count = 0;
    std::shared_ptr<ov::Node> folded_softmax;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            ++transpose_count;
        }
        if (ov::as_type_ptr<ov::op::v8::Softmax>(node) || ov::as_type_ptr<ov::op::v1::Softmax>(node)) {
            folded_softmax = node;
        }
    }

    EXPECT_EQ(transpose_count, 0);
    ASSERT_TRUE(folded_softmax);
    if (auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(folded_softmax)) {
        EXPECT_EQ(softmax_v8->get_axis(), 2);
    } else {
        auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(folded_softmax);
        ASSERT_TRUE(softmax_v1);
        EXPECT_EQ(softmax_v1->get_axis(), 2);
    }
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(folded_softmax->input_value(0).get_node_shared_ptr()));
    EXPECT_EQ(folded_softmax->get_output_partial_shape(0), ov::PartialShape({1, 4, 16, 8400}));
}

TEST(GfxTransforms, FoldTransposeSoftmaxTransposeToSingleTranspose) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16, 8400});
    auto perm0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::op::v1::Transpose>(input, perm0);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(transpose0, 3);
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 2, 1});
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(softmax, perm1);
    auto result = std::make_shared<ov::op::v0::Result>(transpose1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "softmax_transpose_fold_single");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    int transpose_count = 0;
    std::shared_ptr<ov::Node> folded_softmax;
    std::shared_ptr<ov::op::v1::Transpose> folded_transpose;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (auto tr = ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            ++transpose_count;
            folded_transpose = tr;
        }
        if (ov::as_type_ptr<ov::op::v8::Softmax>(node) || ov::as_type_ptr<ov::op::v1::Softmax>(node)) {
            folded_softmax = node;
        }
    }

    ASSERT_TRUE(folded_softmax);
    ASSERT_TRUE(folded_transpose);
    EXPECT_EQ(transpose_count, 1);
    if (auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(folded_softmax)) {
        EXPECT_EQ(softmax_v8->get_axis(), 2);
    } else {
        auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(folded_softmax);
        ASSERT_TRUE(softmax_v1);
        EXPECT_EQ(softmax_v1->get_axis(), 2);
    }
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(folded_softmax->input_value(0).get_node_shared_ptr()));
    EXPECT_EQ(folded_transpose->get_output_partial_shape(0), ov::PartialShape({1, 16, 4, 8400}));
}

TEST(GfxTransforms, SinkTransposeThroughReluAndEliminatePair) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 4, 16, 8400});
    auto perm0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::op::v1::Transpose>(input, perm0);
    auto relu = std::make_shared<ov::op::v0::Relu>(transpose0);
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(relu, perm1);
    auto result = std::make_shared<ov::op::v0::Result>(transpose1);
    auto model =
        std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "transpose_relu_transpose");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    int transpose_count = 0;
    std::shared_ptr<ov::op::v0::Relu> folded_relu;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            ++transpose_count;
        }
        if (auto relu_node = ov::as_type_ptr<ov::op::v0::Relu>(node)) {
            folded_relu = relu_node;
        }
    }

    EXPECT_EQ(transpose_count, 0);
    ASSERT_TRUE(folded_relu);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(folded_relu->input_value(0).get_node_shared_ptr()));
    EXPECT_EQ(folded_relu->get_output_partial_shape(0), ov::PartialShape({1, 4, 16, 8400}));
}

TEST(GfxTransforms, DeduplicateEquivalentTransposeReshapeBranches) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{4, 2, 32, 400});

    auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 40, 40, 64});

    auto transpose0 = std::make_shared<ov::op::v1::Transpose>(input, perm);
    auto reshape0 = std::make_shared<ov::op::v1::Reshape>(transpose0, reshape_shape, true);
    auto relu0 = std::make_shared<ov::op::v0::Relu>(reshape0);

    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(input, perm);
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(transpose1, reshape_shape, true);
    auto softmax1 = std::make_shared<ov::op::v8::Softmax>(reshape1, 3);

    auto result0 = std::make_shared<ov::op::v0::Result>(relu0);
    auto result1 = std::make_shared<ov::op::v0::Result>(softmax1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{input}, "dedup_transpose_reshape");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    int transpose_count = 0;
    int reshape_count = 0;
    std::shared_ptr<ov::op::v1::Reshape> shared_reshape;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            ++transpose_count;
        }
        if (auto reshape_node = ov::as_type_ptr<ov::op::v1::Reshape>(node)) {
            ++reshape_count;
            shared_reshape = reshape_node;
        }
    }

    EXPECT_EQ(transpose_count, 1);
    EXPECT_EQ(reshape_count, 1);
    ASSERT_TRUE(shared_reshape);
}

TEST(GfxTransforms, FoldDflSoftmaxExpectationToMatMul) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 64, 8400});
    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 4, 16, 8400});
    auto reshape0 = std::make_shared<ov::op::v1::Reshape>(input, reshape_shape, true);

    auto perm0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::op::v1::Transpose>(reshape0, perm0);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(transpose0, 3);
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 2, 1});
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(softmax, perm1);

    std::vector<float> weights_data(16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>(i);
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16, 1, 1}, weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(transpose1,
                                                           weights,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{0, 0},
                                                           ov::CoordinateDiff{0, 0},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT);
    auto final_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 4, 8400});
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(conv, final_shape, true);
    auto result = std::make_shared<ov::op::v0::Result>(reshape1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input}, "dfl_expectation_fold");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    int transpose_count = 0;
    int conv_count = 0;
    int matmul_count = 0;
    std::shared_ptr<ov::Node> folded_softmax;
    for (const auto& node : transformed->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Transpose>(node)) {
            ++transpose_count;
        }
        if (ov::as_type_ptr<ov::op::v1::Convolution>(node)) {
            ++conv_count;
        }
        if (ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
            ++matmul_count;
        }
        if (ov::as_type_ptr<ov::op::v8::Softmax>(node) || ov::as_type_ptr<ov::op::v1::Softmax>(node)) {
            folded_softmax = node;
        }
    }

    EXPECT_EQ(transpose_count, 2);
    EXPECT_EQ(conv_count, 0);
    EXPECT_EQ(matmul_count, 1);
    ASSERT_TRUE(folded_softmax);
    if (auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(folded_softmax)) {
        EXPECT_EQ(softmax_v8->get_axis(), 3);
    } else {
        auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(folded_softmax);
        ASSERT_TRUE(softmax_v1);
        EXPECT_EQ(softmax_v1->get_axis(), 3);
    }
    EXPECT_EQ(transformed->output(0).get_partial_shape(), ov::PartialShape({1, 4, 8400}));
}

TEST(GfxTransforms, FoldDflSoftmaxExpectationMatMulPreservesValues) {
    std::vector<float> input_data(1 * 64 * 8);
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>((static_cast<int>(i % 13) - 6) * 0.25f);
    }
    auto input = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 64, 8}, input_data);
    auto reshape_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 4, 16, 8});
    auto reshape0 = std::make_shared<ov::op::v1::Reshape>(input, reshape_shape, true);

    auto perm0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
    auto transpose0 = std::make_shared<ov::op::v1::Transpose>(reshape0, perm0);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(transpose0, 3);
    auto perm1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 2, 1});
    auto transpose1 = std::make_shared<ov::op::v1::Transpose>(softmax, perm1);

    std::vector<float> weights_data(16);
    for (size_t i = 0; i < weights_data.size(); ++i) {
        weights_data[i] = static_cast<float>(i);
    }
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 16, 1, 1}, weights_data);
    auto conv = std::make_shared<ov::op::v1::Convolution>(transpose1,
                                                           weights,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{0, 0},
                                                           ov::CoordinateDiff{0, 0},
                                                           ov::Strides{1, 1},
                                                           ov::op::PadType::EXPLICIT);
    auto final_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 4, 8});
    auto reshape1 = std::make_shared<ov::op::v1::Reshape>(conv, final_shape, true);
    auto result = std::make_shared<ov::op::v0::Result>(reshape1);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{}, "dfl_expectation_values");

    auto transformed = ov::gfx_plugin::transforms::run_pipeline(model);
    ASSERT_TRUE(transformed);

    const auto expected = infer_with_template(model);
    const auto actual = infer_with_template(transformed);
    ASSERT_EQ(expected.get_shape(), actual.get_shape());

    const auto* expected_data = expected.data<const float>();
    const auto* actual_data = actual.data<const float>();
    ASSERT_NE(expected_data, nullptr);
    ASSERT_NE(actual_data, nullptr);
    for (size_t i = 0; i < expected.get_size(); ++i) {
        EXPECT_NEAR(expected_data[i], actual_data[i], 1e-5f) << "index=" << i;
    }
}
