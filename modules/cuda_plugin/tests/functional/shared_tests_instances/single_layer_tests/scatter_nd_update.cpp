// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/scatter_ND_update.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "benchmark.hpp"

using namespace LayerTestsDefinitions;

namespace {

class UniqueComplexIndexGenerator {
public:
    UniqueComplexIndexGenerator(const std::vector<size_t>& shape) {
        distribs_.reserve(shape.size());
        for (const auto& size : shape) distribs_.emplace_back(0, size - 1);
    }

    std::vector<size_t> get_next_complex_index() {
        std::vector<size_t> complexIndex(distribs_.size(), 0);
        while (true) {
            generate_index(complexIndex);
            if (unique(complexIndex)) {
                waste_.emplace(get_hash(complexIndex));
                break;
            }
        }

        return complexIndex;
    }

private:
    void generate_index(std::vector<size_t>& complexIndex) {
        for (size_t i{}; i < distribs_.size(); ++i) complexIndex[i] = distribs_[i](randomEngine_);
    }

    bool unique(const std::vector<size_t>& complexIndex) { return waste_.find(get_hash(complexIndex)) == waste_.end(); }

    std::string get_hash(const std::vector<size_t>& complexIndex) {
        std::ostringstream hash;
        for (const auto& index : complexIndex) hash << index << '.';

        return hash.str();
    }

private:
    std::random_device randomDevice_{};
    std::default_random_engine randomEngine_{randomDevice_()};
    std::vector<std::uniform_int_distribution<size_t>> distribs_;

    std::unordered_set<std::string> waste_;
};
void generate_indices(sliceSelectInShape& slice) {
    const auto& input_shape = std::get<0>(slice);
    const auto& indices_shape = std::get<1>(slice);

    const size_t complex_index_size = indices_shape.back();

    UniqueComplexIndexGenerator gen{{input_shape.cbegin(), input_shape.cbegin() + complex_index_size}};

    const auto indices_size = ngraph::shape_size(indices_shape);
    auto& indices = std::get<2>(slice);
    indices.resize(indices_size);
    for (size_t stride{}; stride < indices_size; stride += complex_index_size) {
        const auto& complex_index = gen.get_next_complex_index();
        for (size_t i{}; i < complex_index.size(); ++i) indices[stride + i] = complex_index[i];
    }
}

}  // namespace

namespace LayerTestsDefinitions {

// The class was inherited to change combineShapes() method
// to enable of indices values generation for large size indices tensors
class CudaScatterNDUpdateLayerTest : public ScatterNDUpdateLayerTest {
public:
    static std::vector<sliceSelectInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>>& inputShapes) {
        auto slices = ScatterNDUpdateLayerTest::combineShapes(inputShapes);
        for (auto& slice : slices) {
            auto& indices = std::get<2>(slice);
            if (indices.empty()) generate_indices(slice);
        }

        return slices;
    }
};

TEST_P(CudaScatterNDUpdateLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
};

}  // namespace LayerTestsDefinitions

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::I32};

const std::vector<InferenceEngine::Precision> idxPrecisions = {InferenceEngine::Precision::I32,
                                                               InferenceEngine::Precision::I64};

// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> smoke_shapes{
    {{4, 3, 2, 3, 2}, {{{2, 2, 1}, {3, 2, 0, 1}}}},
    {{10, 9, 9, 11}, {{{4, 1}, {1, 3, 5, 7}}, {{1, 2}, {4, 6}}, {{2, 3}, {0, 1, 1, 2, 2, 2}}, {{1, 4}, {5, 5, 4, 9}}}},
    {{10, 9, 12, 10, 11}, {{{2, 2, 1}, {5, 6, 2, 8}}, {{2, 3}, {0, 4, 6, 5, 7, 1}}}},
    {{15}, {{{2, 1}, {1, 3}}}},
    {{15, 14}, {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}}},
    {{15, 14, 13}, {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}}},
    {{15, 14, 13, 12},
     {{{2, 1}, {1, 3}},
      {{2, 2}, {2, 3, 10, 11}},
      {{2, 3}, {2, 3, 1, 8, 10, 11}},
      {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
      {{2, 2, 2}, {2, 3, 1, 8, 7, 5, 6, 5}}}},
    {{15, 14, 13, 12, 16},
     {{{2, 1}, {1, 3}},
      {{2, 2}, {2, 3, 10, 11}},
      {{2, 3}, {2, 3, 1, 8, 10, 11}},
      {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
      {{2, 5}, {2, 3, 1, 8, 6, 9, 7, 5, 6, 5}}}},
    {{15, 14, 13, 12, 16, 10},
     {{{2, 1}, {1, 3}},
      {{2, 2}, {2, 3, 10, 11}},
      {{2, 3}, {2, 3, 1, 8, 10, 11}},
      {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
      {{1, 2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
      {{2, 5}, {2, 3, 1, 8, 6, 9, 7, 5, 6, 5}},
      {{2, 6}, {2, 3, 1, 8, 6, 5, 9, 7, 5, 6, 5, 7}}}}};

INSTANTIATE_TEST_CASE_P(
    smoke_ScatterNDUpdate,
    CudaScatterNDUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(CudaScatterNDUpdateLayerTest::combineShapes(smoke_shapes)),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    CudaScatterNDUpdateLayerTest::getTestCaseName);

// map<inputShape map<indicesShape, emptyIndicesValue>>
// updateShape is gotten from inputShape and indicesShape
// emptyIndicesValue will be generated inside CudaScatterNDUpdateLayerTest::combineShapes method
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> smoke_shapes_index_generation{
    {{4, 3, 2, 3, 2}, {{{2, 2, 1}, {}}}},
    {{10, 9, 9, 11}, {{{4, 1}, {}}, {{1, 2}, {}}, {{2, 3}, {}}, {{1, 4}, {}}}},
    {{10, 9, 12, 10, 11}, {{{2, 2, 1}, {}}, {{2, 3}, {}}}},
    {{15}, {{{2, 1}, {}}}},
    {{15, 14}, {{{2, 1}, {}}, {{2, 2}, {}}}},
    {{15, 14, 13}, {{{2, 1}, {}}, {{2, 2}, {}}, {{2, 3}, {}}}},
    {{15, 14, 13, 12}, {{{2, 1}, {}}, {{2, 2}, {}}, {{2, 3}, {}}, {{2, 4}, {}}, {{2, 2, 2}, {}}}},
    {{15, 14, 13, 12, 16}, {{{2, 1}, {}}, {{2, 2}, {}}, {{2, 3}, {}}, {{2, 4}, {}}, {{2, 5}, {}}}},
    {{15, 14, 13, 12, 16, 10},
     {{{2, 1}, {}}, {{2, 2}, {}}, {{2, 3}, {}}, {{2, 4}, {}}, {{1, 2, 4}, {}}, {{2, 5}, {}}, {{2, 6}, {}}}}};

INSTANTIATE_TEST_CASE_P(
    smoke_index_generation_ScatterNDUpdate,
    CudaScatterNDUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(CudaScatterNDUpdateLayerTest::combineShapes(smoke_shapes_index_generation)),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    CudaScatterNDUpdateLayerTest::getTestCaseName);

// yolov5b6 shapes
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> yolov5b6_shapes{
    {{1, 3, 40, 40, 85}, {{{1, 3, 40, 40, 2, 5}, {}}}},
    {{1, 3, 80, 80, 85}, {{{1, 3, 80, 80, 2, 5}, {}}}},
    {{1, 3, 20, 20, 85}, {{{1, 3, 20, 20, 2, 5}, {}}}}};

INSTANTIATE_TEST_CASE_P(
    yolov5b6_ScatterNDUpdate,
    CudaScatterNDUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(CudaScatterNDUpdateLayerTest::combineShapes(yolov5b6_shapes)),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    CudaScatterNDUpdateLayerTest::getTestCaseName);

namespace benchmark {

struct CudaScatterNDUpdateLayerBenchmarkTest : BenchmarkLayerTest<CudaScatterNDUpdateLayerTest> {};

TEST_P(CudaScatterNDUpdateLayerBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("ScatterNDUpdate");
}

// yolov5b6 shapes
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> benchmark_shapes{
    {{100, 1048576}, {{{10, 1}, {1, 4, 9, 11, 13, 17, 19, 23, 27, 37}}}},
    {{1, 3, 80, 80, 85}, {{{1, 3, 80, 80, 2, 5}, {}}}},
    {{15, 14, 13, 12, 16, 10}, {{{2, 1}, {1, 3}}}}};

INSTANTIATE_TEST_CASE_P(
    ScatterNDUpdate_Benchmark,
    CudaScatterNDUpdateLayerBenchmarkTest,
    ::testing::Combine(::testing::ValuesIn(CudaScatterNDUpdateLayerTest::combineShapes(benchmark_shapes)),
                       ::testing::Values(InferenceEngine::Precision::FP32),
                       ::testing::Values(InferenceEngine::Precision::I64),
                       ::testing::Values(CommonTestUtils::DEVICE_CUDA)),
    CudaScatterNDUpdateLayerTest::getTestCaseName);
}  // namespace benchmark
}  // namespace
