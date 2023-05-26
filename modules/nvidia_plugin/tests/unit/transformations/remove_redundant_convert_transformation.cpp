// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include "transformer/remove_redundant_convert_transformation.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::opset10;

using namespace ngraph;
using namespace std;

TEST(remove_redundant_convert, remove_redundant_convert) {
    std::shared_ptr<Function> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::Parameter>(element::f32, shape);
        auto convert_f32 = make_shared<op::v0::Convert>(input_fp32, element::f32);
        f = make_shared<Function>(make_shared<op::v0::Abs>(convert_f32), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(remove_redundant_convert, remove_symmetric_converts) {
    std::shared_ptr<Function> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::Parameter>(element::f32, shape);
        auto convert_f16 = make_shared<op::v0::Convert>(input_fp32, element::f16);
        auto convert_f32 = make_shared<op::v0::Convert>(convert_f16, element::f32);
        f = make_shared<Function>(make_shared<op::v0::Abs>(convert_f32), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(remove_redundant_convert, remove_several_converts) {
    std::shared_ptr<Function> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::Parameter>(element::f32, shape);
        auto convert_f16 = make_shared<op::v0::Convert>(input_fp32, element::f16);
        auto convert_f64 = make_shared<op::v0::Convert>(convert_f16, element::f64);
        auto convert_f32 = make_shared<op::v0::Convert>(convert_f64, element::f32);
        f = make_shared<Function>(make_shared<op::v0::Abs>(convert_f32), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::InitNodeInfo>();
    pass_manager.register_pass<ov::nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}
