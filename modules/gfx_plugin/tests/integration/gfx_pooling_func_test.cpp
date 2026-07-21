// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/openvino.hpp"
#include "shared_tests_instances/test_utils.hpp"

namespace {

enum class PoolingKind {
    Max,
    Average,
};

struct PoolingFunctionalCase {
    std::string name;
    PoolingKind kind;
    ov::element::Type element_type;
    ov::Shape input_shape;
};

class PoolingFunctionalContract {
public:
    explicit PoolingFunctionalContract(PoolingFunctionalCase test_case)
        : test_case_(std::move(test_case)) {}

    std::shared_ptr<ov::Model> make_model() const {
        auto input = std::make_shared<ov::op::v0::Parameter>(test_case_.element_type, test_case_.input_shape);
        ov::Strides strides{2, 2};
        ov::Shape pads{0, 0};
        ov::Shape kernel{2, 2};

        std::shared_ptr<ov::Node> pool;
        if (test_case_.kind == PoolingKind::Max) {
            pool = std::make_shared<ov::op::v1::MaxPool>(input,
                                                         strides,
                                                         pads,
                                                         pads,
                                                         kernel,
                                                         ov::op::RoundingType::FLOOR);
        } else {
            pool = std::make_shared<ov::op::v1::AvgPool>(input,
                                                         strides,
                                                         pads,
                                                         pads,
                                                         kernel,
                                                         true,
                                                         ov::op::RoundingType::FLOOR);
        }

        auto result = std::make_shared<ov::op::v0::Result>(pool);
        return std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{input},
                                           test_case_.name);
    }

    ov::Tensor make_input() const {
        ov::Tensor input{test_case_.element_type, test_case_.input_shape};
        const auto count = input.get_size();
        if (test_case_.element_type == ov::element::f16) {
            auto* data = input.data<ov::float16>();
            for (std::size_t i = 0; i < count; ++i) {
                data[i] = ov::float16(sample_value(i));
            }
        } else {
            auto* data = input.data<float>();
            for (std::size_t i = 0; i < count; ++i) {
                data[i] = sample_value(i);
            }
        }
        return input;
    }

    float abs_tolerance() const {
        return test_case_.element_type == ov::element::f16 ? 2e-3f : 1e-5f;
    }

    float rel_tolerance() const {
        return test_case_.element_type == ov::element::f16 ? 2e-3f : 1e-5f;
    }

private:
    static float sample_value(std::size_t index) {
        const auto lane = static_cast<int>(index % 19);
        return static_cast<float>(lane - 9) * 0.25f;
    }

    PoolingFunctionalCase test_case_;
};

class GfxPoolingFunctionalTest : public testing::TestWithParam<PoolingFunctionalCase>,
                                 protected ov::test::utils::GfxVsTemplateFixture {
protected:
    void compare_with_template() {
        PoolingFunctionalContract contract(GetParam());
        compare_gfx_vs_template(contract.make_model(),
                                std::vector<ov::Tensor>{contract.make_input()},
                                {},
                                contract.abs_tolerance(),
                                contract.rel_tolerance());
    }
};

TEST_P(GfxPoolingFunctionalTest, MatchesTemplate) {
    compare_with_template();
}

std::vector<PoolingFunctionalCase> pooling_cases() {
    return {
        {"maxpool2d_f32_c1", PoolingKind::Max, ov::element::f32, ov::Shape{1, 1, 4, 4}},
        {"avgpool2d_f32_c1", PoolingKind::Average, ov::element::f32, ov::Shape{1, 1, 4, 4}},
        {"maxpool2d_f16_c3", PoolingKind::Max, ov::element::f16, ov::Shape{1, 3, 4, 4}},
        {"avgpool2d_f16_c3", PoolingKind::Average, ov::element::f16, ov::Shape{1, 3, 4, 4}},
    };
}

std::string pooling_case_name(const testing::TestParamInfo<PoolingFunctionalCase>& info) {
    return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(GfxPooling,
                         GfxPoolingFunctionalTest,
                         testing::ValuesIn(pooling_cases()),
                         pooling_case_name);

}  // namespace
