// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_layout.hpp"
#include "conv_arm.hpp"
#include "pool_arm.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace ov;
using namespace ArmPlugin;

TEST_F(TransformationTestsF, ConvertLayoutArmConvolution2D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto weights = opset8::Constant::create(element::f32, Shape{5, 3, 7, 7}, {2});
        auto conv = std::make_shared<opset::ArmConvolution>(param, weights, Strides{1, 1},
                                                            CoordinateDiff{1, 1}, CoordinateDiff{0, 0},
                                                            Strides{1, 1}, op::PadType::EXPLICIT);
        model = std::make_shared<Model>(conv, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto weights = opset8::Constant::create(element::f32, Shape{5, 3, 7, 7}, {2});
        auto transpose_on_input = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1}));
        auto transpose_on_weights = std::make_shared<opset::Transpose>(weights, opset8::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1}));
        auto conv = std::make_shared<opset::ArmConvolution>(transpose_on_input, transpose_on_weights, Strides{1, 1},
                                                            CoordinateDiff{1, 1}, CoordinateDiff{0, 0},
                                                            Strides{1, 1}, op::PadType::EXPLICIT, PartialShape{1, 9, 9, 5});
        auto transpose_on_output = std::make_shared<opset::Transpose>(conv, opset8::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2}));
        model_ref = std::make_shared<Model>(transpose_on_output, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmConvolution3D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto weights = opset8::Constant::create(element::f32, Shape{5, 3, 7, 7, 7}, {2});
        auto conv = std::make_shared<opset::ArmConvolution>(param, weights, Strides{1, 1, 1},
                                                            CoordinateDiff{1, 1, 1}, CoordinateDiff{0, 0, 0},
                                                            Strides{1, 1, 1}, op::PadType::EXPLICIT);
        model = std::make_shared<Model>(conv, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto weights = opset8::Constant::create(element::f32, Shape{5, 3, 7, 7, 7}, {2});
        auto transpose_on_input = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{5}, {0, 2, 3, 4, 1}));
        auto transpose_on_weights = std::make_shared<opset::Transpose>(weights, opset8::Constant::create(element::i32, Shape{5}, {0, 2, 3, 4, 1}));
        auto conv = std::make_shared<opset::ArmConvolution>(transpose_on_input, transpose_on_weights, Strides{1, 1, 1},
                                                            CoordinateDiff{1, 1, 1}, CoordinateDiff{0, 0, 0},
                                                            Strides{1, 1, 1}, op::PadType::EXPLICIT, PartialShape{1, 9, 9, 9, 5});
        auto transpose_on_output = std::make_shared<opset::Transpose>(conv, opset8::Constant::create(element::i32, Shape{5}, {0, 4, 1, 2, 3}));
        model_ref = std::make_shared<Model>(transpose_on_output, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmMaxPoolV12D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto pool = std::make_shared<opset::v1::ArmMaxPool>(param, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{2, 2});
        model = std::make_shared<Model>(pool, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto transpose = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1}));
        auto pool = std::make_shared<opset::v1::ArmMaxPool>(transpose, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{2, 2},
                                                            op::RoundingType::FLOOR, op::PadType::EXPLICIT, PartialShape{1, 14, 14, 3});
        auto transpose_on_output = std::make_shared<opset::Transpose>(pool, opset8::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2}));
        model_ref = std::make_shared<Model>(transpose_on_output, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmMaxPoolV13D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto pool = std::make_shared<opset::v1::ArmMaxPool>(param, Strides{1, 1, 1}, Shape{1, 1, 1}, Shape{0, 0, 0}, Shape{2, 2, 2});
        model = std::make_shared<Model>(pool, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto transpose = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{5}, {0, 2, 3, 4, 1}));
        auto pool = std::make_shared<opset::v1::ArmMaxPool>(transpose, Strides{1, 1, 1}, Shape{1, 1, 1}, Shape{0, 0, 0}, Shape{2, 2, 2},
                                                            op::RoundingType::FLOOR, op::PadType::EXPLICIT, PartialShape{1, 14, 14, 14, 3});
        auto transpose_on_output = std::make_shared<opset::Transpose>(pool, opset8::Constant::create(element::i32, Shape{5}, {0, 4, 1, 2, 3}));
        model_ref = std::make_shared<Model>(transpose_on_output, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmMaxPoolV82D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto pool = std::make_shared<opset::v8::ArmMaxPool>(param, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{1, 1}, Shape{2, 2});
        auto output = std::make_shared<opset8::Result>(pool->output(0));
        auto indexes = std::make_shared<opset8::Result>(pool->output(1));
        model = std::make_shared<Model>(ResultVector{output, indexes}, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto transpose = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1}));
        auto pool = std::make_shared<opset::v8::ArmMaxPool>(transpose, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{1, 1}, Shape{2, 2},
                                                            op::RoundingType::FLOOR, op::PadType::EXPLICIT, element::i64, 0, PartialShape{1, 14, 14, 3});
        auto transpose_on_output = std::make_shared<opset::Transpose>(pool->output(0), opset8::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2}));
        auto transpose_on_indexes = std::make_shared<opset::Transpose>(pool->output(1), opset8::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2}));
        model_ref = std::make_shared<Model>(NodeVector{transpose_on_output, transpose_on_indexes}, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmMaxPoolV83D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto pool = std::make_shared<opset::v8::ArmMaxPool>(param, Strides{1, 1, 1}, Shape{1, 1, 1}, Shape{0, 0, 0}, Shape{1, 1, 1}, Shape{2, 2, 2});
        model = std::make_shared<Model>(pool, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto transpose = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{5}, {0, 2, 3, 4, 1}));
        auto pool = std::make_shared<opset::v8::ArmMaxPool>(transpose, Strides{1, 1, 1}, Shape{1, 1, 1}, Shape{0, 0, 0}, Shape{1, 1, 1}, Shape{2, 2, 2},
                                                            op::RoundingType::FLOOR, op::PadType::EXPLICIT, element::i64, 0, PartialShape{1, 14, 14, 14, 3});
        auto transpose_on_output = std::make_shared<opset::Transpose>(pool->output(0), opset8::Constant::create(element::i32, Shape{5}, {0, 4, 1, 2, 3}));
        auto transpose_on_indexes = std::make_shared<opset::Transpose>(pool->output(1), opset8::Constant::create(element::i32, Shape{5}, {0, 4, 1, 2, 3}));
        model_ref = std::make_shared<Model>(NodeVector{transpose_on_output, transpose_on_indexes}, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmAvgPool2D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto pool = std::make_shared<opset::v1::ArmAvgPool>(param, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{2, 2}, true);
        model = std::make_shared<Model>(pool, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14});
        auto transpose = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{4}, {0, 2, 3, 1}));
        auto pool = std::make_shared<opset::v1::ArmAvgPool>(transpose, Strides{1, 1}, Shape{1, 1}, Shape{0, 0}, Shape{2, 2}, true,
                                                            op::RoundingType::FLOOR, op::PadType::EXPLICIT, PartialShape{1, 14, 14, 3});
        auto transpose_on_output = std::make_shared<opset::Transpose>(pool, opset8::Constant::create(element::i32, Shape{4}, {0, 3, 1, 2}));
        model_ref = std::make_shared<Model>(transpose_on_output, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLayoutArmAvgPool3D) {
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto pool = std::make_shared<opset::v1::ArmAvgPool>(param, Strides{1, 1, 1}, Shape{1, 1, 1}, Shape{0, 0, 0}, Shape{2, 2, 2}, true);
        model = std::make_shared<Model>(pool, ParameterVector{param});
    }
    {
        auto param = std::make_shared<opset8::Parameter>(element::f32, Shape{1, 3, 14, 14, 14});
        auto transpose = std::make_shared<opset::Transpose>(param, opset8::Constant::create(element::i32, Shape{5}, {0, 2, 3, 4, 1}));
        auto pool = std::make_shared<opset::v1::ArmAvgPool>(transpose, Strides{1, 1, 1}, Shape{1, 1, 1}, Shape{0, 0, 0}, Shape{2, 2, 2}, true,
                                                            op::RoundingType::FLOOR, op::PadType::EXPLICIT, PartialShape{1, 14, 14, 14, 3});
        auto transpose_on_output = std::make_shared<opset::Transpose>(pool, opset8::Constant::create(element::i32, Shape{5}, {0, 4, 1, 2, 3}));
        model_ref = std::make_shared<Model>(transpose_on_output, ParameterVector{param});
    }

    manager.register_pass<ArmPlugin::pass::ConvertLayout>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
