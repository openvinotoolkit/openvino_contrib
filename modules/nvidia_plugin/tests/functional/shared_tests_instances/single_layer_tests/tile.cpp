// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tile.hpp"

#include "cuda_test_constants.hpp"

namespace {
using namespace ov::test;
using namespace ov::test::utils;

// precision, data shape, repeats (may be longer than the data rank)
using TileParams = std::tuple<ov::element::Type, ov::Shape, std::vector<std::int64_t>>;

class TileNVIDIATest : public ov::test::SubgraphBaseTest, public testing::WithParamInterface<TileParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TileParams>& obj) {
        const auto& [precision, data_shape, repeats] = obj.param;
        std::ostringstream result;
        result << "prec=" << precision << "_shape=" << ov::test::utils::vec2str(data_shape) << "_repeats=";
        for (auto r : repeats) {
            result << r << ".";
        }
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = DEVICE_NVIDIA;
        const auto& [precision, data_shape, repeats] = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>{data_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(precision, inputDynamicShapes[0]);
        auto repeats_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{repeats.size()}, repeats);
        auto tile = std::make_shared<ov::op::v0::Tile>(data, repeats_const);
        auto result = std::make_shared<ov::op::v0::Result>(tile);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data}, "Tile");
    }
};

TEST_P(TileNVIDIATest, Inference) {
    run();
}

const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16, ov::element::i32};

INSTANTIATE_TEST_SUITE_P(
    smoke_Tile,
    TileNVIDIATest,
    ::testing::Combine(::testing::ValuesIn(precisions),
                       ::testing::ValuesIn(std::vector<ov::Shape>{{2, 3}, {1, 4}, {2, 3, 4}, {1, 1, 8}}),
                       ::testing::Values(std::vector<std::int64_t>{2, 2, 2})),
    TileNVIDIATest::getTestCaseName);

// Per-axis repeats and rank promotion (repeats longer than the data rank).
INSTANTIATE_TEST_SUITE_P(smoke_Tile_PerAxis,
                         TileNVIDIATest,
                         ::testing::Values(TileParams{ov::element::f32, ov::Shape{2, 3, 4}, {1, 2, 1}},
                                           TileParams{ov::element::f32, ov::Shape{3}, {1, 2, 2}},  // promotion -> [1,2,6]
                                           TileParams{ov::element::f16, ov::Shape{1, 1, 8}, {2, 1, 1}},  // CFG batch dup
                                           TileParams{ov::element::i32, ov::Shape{1, 4}, {3, 1}}),
                         TileNVIDIATest::getTestCaseName);

}  // namespace
