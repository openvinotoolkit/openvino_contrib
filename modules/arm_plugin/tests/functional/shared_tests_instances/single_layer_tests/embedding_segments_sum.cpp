// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/embedding_segments_sum.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
        ov::test::ElementType::f32,
        ov::test::ElementType::f16,
        ov::test::ElementType::i32,
        ov::test::ElementType::i16,
        ov::test::ElementType::u8
};

const std::vector<ov::test::ElementType> indPrecisions = {
        ov::test::ElementType::i64,
        ov::test::ElementType::i32,
};

const std::vector<ov::Shape> emb_table_shape = {{5, 6}, {10, 35}, {5, 4, 16}};
const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}, {4, 4, 3, 1, 2}};
const std::vector<std::vector<size_t>> segment_ids = {{0, 1, 2, 3, 4}, {0, 0, 2, 2, 4}};
const std::vector<size_t> num_segments = {5, 7};
const std::vector<size_t> default_index = {0, 4};
const std::vector<bool> with_weights = {false, true};
const std::vector<bool> with_default_index = {false, true};

const auto embSegmentsSumArgSet = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(emb_table_shape)),
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(segment_ids),
        ::testing::ValuesIn(num_segments),
        ::testing::ValuesIn(default_index),
        ::testing::ValuesIn(with_weights),
        ::testing::ValuesIn(with_default_index)
);

INSTANTIATE_TEST_CASE_P(smoke_EmbeddingSegmentsSum, EmbeddingSegmentsSumLayerTest,
                        ::testing::Combine(
                                embSegmentsSumArgSet,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(indPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        EmbeddingSegmentsSumLayerTest::getTestCaseName);
}  // namespace
