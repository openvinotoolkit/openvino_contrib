// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

using SDPAParams = std::tuple<
    std::vector<InputShape>,  // Q, K, V shapes
    bool,                     // is_causal
    ov::element::Type         // precision
>;

class SDPANVIDIATest : virtual public SubgraphBaseTest,
                        public testing::WithParamInterface<SDPAParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SDPAParams>& obj) {
        const auto& [input_shapes, is_causal, precision] = obj.param;

        std::ostringstream result;
        result << "Q=" << partialShape2str({input_shapes[0].first}) << "_";
        result << "K=" << partialShape2str({input_shapes[1].first}) << "_";
        result << "V=" << partialShape2str({input_shapes[2].first}) << "_";
        result << "TS=";
        for (size_t i = 0; i < input_shapes[0].second.size(); i++) {
            result << "{";
            for (size_t j = 0; j < input_shapes.size(); j++) {
                result << vec2str(input_shapes[j].second[i]);
                if (j < input_shapes.size() - 1) result << "_";
            }
            result << "}_";
        }
        result << "causal=" << (is_causal ? "true" : "false") << "_";
        result << "precision=" << precision;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [input_shapes, is_causal, precision] = GetParam();
        targetDevice = DEVICE_NVIDIA;

        init_input_shapes(input_shapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }

        auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(
            params[0], params[1], params[2], is_causal);
        auto result = std::make_shared<ov::op::v0::Result>(sdpa);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, params, "SDPA");

        if (precision == ov::element::f16) {
            abs_threshold = 0.025;
            rel_threshold = 0.025;
        }
    }
};

TEST_P(SDPANVIDIATest, Inference) {
    run();
}

// 4D: [batch, heads, seq_len, head_dim]
const std::vector<std::vector<InputShape>> shapes_4d_static = {
    // Small: batch=1, heads=1, seq=4, dim=8
    {
        {{1, 1, 4, 8}, {{1, 1, 4, 8}}},
        {{1, 1, 4, 8}, {{1, 1, 4, 8}}},
        {{1, 1, 4, 8}, {{1, 1, 4, 8}}},
    },
    // Medium: batch=1, heads=4, seq=16, dim=32
    {
        {{1, 4, 16, 32}, {{1, 4, 16, 32}}},
        {{1, 4, 16, 32}, {{1, 4, 16, 32}}},
        {{1, 4, 16, 32}, {{1, 4, 16, 32}}},
    },
    // Typical LLM: batch=1, heads=8, seq=32, dim=64
    {
        {{1, 8, 32, 64}, {{1, 8, 32, 64}}},
        {{1, 8, 32, 64}, {{1, 8, 32, 64}}},
        {{1, 8, 32, 64}, {{1, 8, 32, 64}}},
    },
};

// 3D: [seq_len, head_count, head_dim]
const std::vector<std::vector<InputShape>> shapes_3d_static = {
    {
        {{4, 2, 8}, {{4, 2, 8}}},
        {{4, 2, 8}, {{4, 2, 8}}},
        {{4, 2, 8}, {{4, 2, 8}}},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_SDPA_4D_f32,
                         SDPANVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(shapes_4d_static),
                                            ::testing::Values(false, true),
                                            ::testing::Values(ov::element::f32)),
                         SDPANVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SDPA_4D_f16,
                         SDPANVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(shapes_4d_static),
                                            ::testing::Values(false, true),
                                            ::testing::Values(ov::element::f16)),
                         SDPANVIDIATest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SDPA_3D_f32,
                         SDPANVIDIATest,
                         ::testing::Combine(::testing::ValuesIn(shapes_3d_static),
                                            ::testing::Values(false, true),
                                            ::testing::Values(ov::element::f32)),
                         SDPANVIDIATest::getTestCaseName);

}  // namespace
