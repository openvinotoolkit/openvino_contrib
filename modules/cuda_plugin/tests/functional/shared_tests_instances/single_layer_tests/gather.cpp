// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gather.hpp"

#include <fmt/format.h>

#include <cuda_graph.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_profiler.hpp>
#include <cuda_test_constants.hpp>
#include <error.hpp>

namespace LayerTestsDefinitions {

// The class was inherited to change getTestCaseName() method
// to disable printing of the indices tensors because of their large size in the tests
// printing of them sometimes leads to seg fault
class CudaGatherLayerTest : public GatherLayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gatherParamsTuple>& obj) {
        int axis;
        std::vector<int> indices;
        std::vector<size_t> indicesShape, inputShape;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        std::tie(
            indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) =
            obj.param;
        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
        result << "axis=" << axis << "_";
        result << "indicesShape=" << CommonTestUtils::vec2str(indicesShape) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "inPRC=" << inPrc.name() << "_";
        result << "outPRC=" << outPrc.name() << "_";
        result << "inL=" << inLayout << "_";
        result << "outL=" << outLayout << "_";
        result << "trgDev=" << targetName << "_";
        return result.str();
    }
};

TEST_P(CudaGatherLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    gatherParamsTuple params = GetParam();
    inPrc = std::get<5>(params);
    outPrc = std::get<6>(params);

    Run();
};

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

struct GatherTestParams {
    std::vector<size_t> params_shape_;
    std::vector<size_t> indices_shape_;
    int axis_ = 0;
    int batch_dims_ = 0;
    std::vector<InferenceEngine::Precision> net_precisions_ = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};
    InferenceEngine::Precision input_precision_ = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision output_precision_ = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Layout input_layout_ = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout output_layout_ = InferenceEngine::Layout::ANY;
    LayerTestsUtils::TargetDevice device_ = CommonTestUtils::DEVICE_CUDA;
};

template <typename T>
std::vector<T> generate_indices(const GatherTestParams& test_params) {
    static std::random_device r_device;
    static std::default_random_engine r_engine{r_device()};

    const auto params_shape_size = test_params.params_shape_.size();
    const auto axis = test_params.axis_;
    const unsigned normalized_axis = axis >= 0 ? axis : axis + params_shape_size;
    if (normalized_axis >= params_shape_size) {
        CUDAPlugin::throwIEException(
            fmt::format("normalized_axis >= params_shape_size: {} >= {}", normalized_axis, params_shape_size));
    }
    std::uniform_int_distribution<T> distr(0, test_params.params_shape_[normalized_axis] - 1);

    const auto indices_size = ngraph::shape_size(test_params.indices_shape_);
    std::vector<T> indices(indices_size);
    std::generate(indices.begin(), indices.end(), [&]() { return distr(r_engine); });
    return indices;
}

const GatherTestParams smoke_01_params_v1_v7 = {{4, 3}, {2}};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v1_01,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_01_params_v1_v7)),
                                           ::testing::Values(smoke_01_params_v1_v7.indices_shape_),
                                           ::testing::Values(smoke_01_params_v1_v7.axis_),
                                           ::testing::Values(smoke_01_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(smoke_01_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_01_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_01_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_01_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_01_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_01_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_01,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_01_params_v1_v7.params_shape_),
                                           ::testing::Values(smoke_01_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_01_params_v1_v7.axis_,
                                                                             smoke_01_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_01_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_01_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_01_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_01_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_01_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_01_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_02_params_v1_v7 = {{1, 2, 3}, {1}, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v1_02,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_02_params_v1_v7)),
                                           ::testing::Values(smoke_02_params_v1_v7.indices_shape_),
                                           ::testing::Values(smoke_02_params_v1_v7.axis_),
                                           ::testing::Values(smoke_02_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(smoke_02_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_02_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_02_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_02_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_02_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_02_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_02,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_02_params_v1_v7.params_shape_),
                                           ::testing::Values(smoke_02_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_02_params_v1_v7.axis_,
                                                                             smoke_02_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_02_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_02_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_02_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_02_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_02_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_02_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_03_params_v1_v7 = {{1, 2, 3}, {7}, -2};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v1_03,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_03_params_v1_v7)),
                                           ::testing::Values(smoke_03_params_v1_v7.indices_shape_),
                                           ::testing::Values(smoke_03_params_v1_v7.axis_),
                                           ::testing::Values(smoke_03_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(smoke_03_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_03_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_03_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_03_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_03_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_03_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_03,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_03_params_v1_v7.params_shape_),
                                           ::testing::Values(smoke_03_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_03_params_v1_v7.axis_,
                                                                             smoke_03_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_03_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_03_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_03_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_03_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_03_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_03_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_04_params_v1_v7 = {{1, 2, 3}, {7, 5}, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v1_04,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_04_params_v1_v7)),
                                           ::testing::Values(smoke_04_params_v1_v7.indices_shape_),
                                           ::testing::Values(smoke_04_params_v1_v7.axis_),
                                           ::testing::Values(smoke_04_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(smoke_04_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_04_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_04_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_04_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_04_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_04_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_04,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_04_params_v1_v7.params_shape_),
                                           ::testing::Values(smoke_04_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_04_params_v1_v7.axis_,
                                                                             smoke_04_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_04_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_04_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_04_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_04_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_04_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_04_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_05_params_v7 = {{3, 1, 2, 3}, {3, 7, 5}, 1, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_05,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_05_params_v7.params_shape_),
                                           ::testing::Values(smoke_05_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_05_params_v7.axis_,
                                                                             smoke_05_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_05_params_v7.net_precisions_),
                                           ::testing::Values(smoke_05_params_v7.input_precision_),
                                           ::testing::Values(smoke_05_params_v7.output_precision_),
                                           ::testing::Values(smoke_05_params_v7.input_layout_),
                                           ::testing::Values(smoke_05_params_v7.output_layout_),
                                           ::testing::Values(smoke_05_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_06_params_v7 = {{3, 2, 3, 1, 2, 3}, {3, 2, 3, 7, 5}, 4, 3};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_06,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_06_params_v7.params_shape_),
                                           ::testing::Values(smoke_06_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_06_params_v7.axis_,
                                                                             smoke_06_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_06_params_v7.net_precisions_),
                                           ::testing::Values(smoke_06_params_v7.input_precision_),
                                           ::testing::Values(smoke_06_params_v7.output_precision_),
                                           ::testing::Values(smoke_06_params_v7.input_layout_),
                                           ::testing::Values(smoke_06_params_v7.output_layout_),
                                           ::testing::Values(smoke_06_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_07_ov_params_v1_v7 = {{1, 5}, {1, 3}};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v1_07,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(smoke_07_ov_params_v1_v7)),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.indices_shape_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.axis_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(smoke_07_ov_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_07,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_07_ov_params_v1_v7.params_shape_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_07_ov_params_v1_v7.axis_,
                                                                             smoke_07_ov_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_07_ov_params_v1_v7.net_precisions_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.input_precision_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.output_precision_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.input_layout_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.output_layout_),
                                           ::testing::Values(smoke_07_ov_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_08_ov_params_v7 = {{2, 5}, {2, 3}, 1, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_08,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_08_ov_params_v7.params_shape_),
                                           ::testing::Values(smoke_08_ov_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_08_ov_params_v7.axis_,
                                                                             smoke_08_ov_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_08_ov_params_v7.net_precisions_),
                                           ::testing::Values(smoke_08_ov_params_v7.input_precision_),
                                           ::testing::Values(smoke_08_ov_params_v7.output_precision_),
                                           ::testing::Values(smoke_08_ov_params_v7.input_layout_),
                                           ::testing::Values(smoke_08_ov_params_v7.output_layout_),
                                           ::testing::Values(smoke_08_ov_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_09_ov_params_v7 = {{2, 2, 5}, {2, 2, 3}, 2, 2};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_09,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_09_ov_params_v7.params_shape_),
                                           ::testing::Values(smoke_09_ov_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_09_ov_params_v7.axis_,
                                                                             smoke_09_ov_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_09_ov_params_v7.net_precisions_),
                                           ::testing::Values(smoke_09_ov_params_v7.input_precision_),
                                           ::testing::Values(smoke_09_ov_params_v7.output_precision_),
                                           ::testing::Values(smoke_09_ov_params_v7.input_layout_),
                                           ::testing::Values(smoke_09_ov_params_v7.output_layout_),
                                           ::testing::Values(smoke_09_ov_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_10_ov_params_v7 = {{2, 1, 5, 4}, {2, 3}, 2, 1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_10,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_10_ov_params_v7.params_shape_),
                                           ::testing::Values(smoke_10_ov_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_10_ov_params_v7.axis_,
                                                                             smoke_10_ov_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_10_ov_params_v7.net_precisions_),
                                           ::testing::Values(smoke_10_ov_params_v7.input_precision_),
                                           ::testing::Values(smoke_10_ov_params_v7.output_precision_),
                                           ::testing::Values(smoke_10_ov_params_v7.input_layout_),
                                           ::testing::Values(smoke_10_ov_params_v7.output_layout_),
                                           ::testing::Values(smoke_10_ov_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams smoke_11_ov_params_v7 = {{2, 5}, {2, 3}, 1, -1};

INSTANTIATE_TEST_CASE_P(smoke_Gather_v7_11,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(smoke_11_ov_params_v7.params_shape_),
                                           ::testing::Values(smoke_11_ov_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(smoke_11_ov_params_v7.axis_,
                                                                             smoke_11_ov_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(smoke_11_ov_params_v7.net_precisions_),
                                           ::testing::Values(smoke_11_ov_params_v7.input_precision_),
                                           ::testing::Values(smoke_11_ov_params_v7.output_precision_),
                                           ::testing::Values(smoke_11_ov_params_v7.input_layout_),
                                           ::testing::Values(smoke_11_ov_params_v7.output_layout_),
                                           ::testing::Values(smoke_11_ov_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

// ------------- Tacotron2 shapes -------------
const GatherTestParams tacotron2_enc_params_v1_v7 = {{148, 512}, {1, 1000}};

INSTANTIATE_TEST_CASE_P(Gather_v1_Tacotron2_shapes_enc,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(tacotron2_enc_params_v1_v7)),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.indices_shape_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.axis_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(tacotron2_enc_params_v1_v7.net_precisions_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.input_precision_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.output_precision_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.input_layout_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.output_layout_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Gather_v7_Tacotron2_shapes_enc,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(tacotron2_enc_params_v1_v7.params_shape_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(tacotron2_enc_params_v1_v7.axis_,
                                                                             tacotron2_enc_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(tacotron2_enc_params_v1_v7.net_precisions_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.input_precision_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.output_precision_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.input_layout_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.output_layout_),
                                           ::testing::Values(tacotron2_enc_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

// ------------- LPCNet shapes -------------
const GatherTestParams lpcnet_enc_params_v1_v7 = {{256, 64}, {64, 64, 1}};

INSTANTIATE_TEST_CASE_P(Gather_v1_LPCnet_shapes_enc,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(lpcnet_enc_params_v1_v7)),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.indices_shape_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.axis_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(lpcnet_enc_params_v1_v7.net_precisions_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.input_precision_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.output_precision_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.input_layout_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.output_layout_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Gather_v7_LPCnet_shapes_enc,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(lpcnet_enc_params_v1_v7.params_shape_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(lpcnet_enc_params_v1_v7.axis_,
                                                                             lpcnet_enc_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(lpcnet_enc_params_v1_v7.net_precisions_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.input_precision_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.output_precision_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.input_layout_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.output_layout_),
                                           ::testing::Values(lpcnet_enc_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams lpcnet_dec_params_v1_v7 = {{256, 128}, {64, 64, 3}};

INSTANTIATE_TEST_CASE_P(Gather_v1_LPCnet_shapes_dec,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(lpcnet_dec_params_v1_v7)),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.indices_shape_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.axis_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(lpcnet_dec_params_v1_v7.net_precisions_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.input_precision_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.output_precision_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.input_layout_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.output_layout_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Gather_v7_LPCnet_shapes_dec,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(lpcnet_dec_params_v1_v7.params_shape_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(lpcnet_dec_params_v1_v7.axis_,
                                                                             lpcnet_dec_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(lpcnet_dec_params_v1_v7.net_precisions_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.input_precision_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.output_precision_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.input_layout_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.output_layout_),
                                           ::testing::Values(lpcnet_dec_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

// ------------- Big shapes -------------
const GatherTestParams ov_params_v1 = {{6, 12, 10, 24}, {15, 4, 20, 28}, 1};

INSTANTIATE_TEST_CASE_P(Gather_v1_OV,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(ov_params_v1)),
                                           ::testing::Values(ov_params_v1.indices_shape_),
                                           ::testing::Values(ov_params_v1.axis_),
                                           ::testing::Values(ov_params_v1.params_shape_),
                                           ::testing::ValuesIn(ov_params_v1.net_precisions_),
                                           ::testing::Values(ov_params_v1.input_precision_),
                                           ::testing::Values(ov_params_v1.output_precision_),
                                           ::testing::Values(ov_params_v1.input_layout_),
                                           ::testing::Values(ov_params_v1.output_layout_),
                                           ::testing::Values(ov_params_v1.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams ov_params_v7 = {{2, 64, 128}, {2, 32, 21}, 1, 1};

INSTANTIATE_TEST_CASE_P(Gather_v7_OV,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(ov_params_v7.params_shape_),
                                           ::testing::Values(ov_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(ov_params_v7.axis_,
                                                                             ov_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(ov_params_v7.net_precisions_),
                                           ::testing::Values(ov_params_v7.input_precision_),
                                           ::testing::Values(ov_params_v7.output_precision_),
                                           ::testing::Values(ov_params_v7.input_layout_),
                                           ::testing::Values(ov_params_v7.output_layout_),
                                           ::testing::Values(ov_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams big_01_params_v1_v7 = {{2048, 32, 2048}, {2, 8, 1, 1}, 0, 0, {InferenceEngine::Precision::FP32}};

INSTANTIATE_TEST_CASE_P(Gather_v1_big_01,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(big_01_params_v1_v7)),
                                           ::testing::Values(big_01_params_v1_v7.indices_shape_),
                                           ::testing::Values(big_01_params_v1_v7.axis_),
                                           ::testing::Values(big_01_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(big_01_params_v1_v7.net_precisions_),
                                           ::testing::Values(big_01_params_v1_v7.input_precision_),
                                           ::testing::Values(big_01_params_v1_v7.output_precision_),
                                           ::testing::Values(big_01_params_v1_v7.input_layout_),
                                           ::testing::Values(big_01_params_v1_v7.output_layout_),
                                           ::testing::Values(big_01_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Gather_v7_big_01,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(big_01_params_v1_v7.params_shape_),
                                           ::testing::Values(big_01_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(big_01_params_v1_v7.axis_,
                                                                             big_01_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(big_01_params_v1_v7.net_precisions_),
                                           ::testing::Values(big_01_params_v1_v7.input_precision_),
                                           ::testing::Values(big_01_params_v1_v7.output_precision_),
                                           ::testing::Values(big_01_params_v1_v7.input_layout_),
                                           ::testing::Values(big_01_params_v1_v7.output_layout_),
                                           ::testing::Values(big_01_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams big_02_params_v1_v7 = {{2048, 32, 2048}, {2, 8, 1, 1}, 2, 0, {InferenceEngine::Precision::FP16}};

INSTANTIATE_TEST_CASE_P(Gather_v1_big_02,
                        CudaGatherLayerTest,
                        ::testing::Combine(::testing::Values(generate_indices<int>(big_02_params_v1_v7)),
                                           ::testing::Values(big_02_params_v1_v7.indices_shape_),
                                           ::testing::Values(big_02_params_v1_v7.axis_),
                                           ::testing::Values(big_02_params_v1_v7.params_shape_),
                                           ::testing::ValuesIn(big_02_params_v1_v7.net_precisions_),
                                           ::testing::Values(big_02_params_v1_v7.input_precision_),
                                           ::testing::Values(big_02_params_v1_v7.output_precision_),
                                           ::testing::Values(big_02_params_v1_v7.input_layout_),
                                           ::testing::Values(big_02_params_v1_v7.output_layout_),
                                           ::testing::Values(big_02_params_v1_v7.device_)),
                        CudaGatherLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Gather_v7_big_02,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(big_02_params_v1_v7.params_shape_),
                                           ::testing::Values(big_02_params_v1_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(big_02_params_v1_v7.axis_,
                                                                             big_02_params_v1_v7.batch_dims_)),
                                           ::testing::ValuesIn(big_02_params_v1_v7.net_precisions_),
                                           ::testing::Values(big_02_params_v1_v7.input_precision_),
                                           ::testing::Values(big_02_params_v1_v7.output_precision_),
                                           ::testing::Values(big_02_params_v1_v7.input_layout_),
                                           ::testing::Values(big_02_params_v1_v7.output_layout_),
                                           ::testing::Values(big_02_params_v1_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams big_03_params_v7 = {
    {2, 1, 512, 32, 256}, {2, 1, 2, 8, 1, 1}, 1, 1, {InferenceEngine::Precision::FP16}};

INSTANTIATE_TEST_CASE_P(Gather_v7_big_03,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(big_03_params_v7.params_shape_),
                                           ::testing::Values(big_03_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(big_03_params_v7.axis_,
                                                                             big_03_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(big_03_params_v7.net_precisions_),
                                           ::testing::Values(big_03_params_v7.input_precision_),
                                           ::testing::Values(big_03_params_v7.output_precision_),
                                           ::testing::Values(big_03_params_v7.input_layout_),
                                           ::testing::Values(big_03_params_v7.output_layout_),
                                           ::testing::Values(big_03_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

const GatherTestParams big_04_params_v7 = {
    {2, 1, 2, 512, 32, 1, 1, 1024}, {2, 1, 2, 1, 2, 8, 1, 1}, 4, 3, {InferenceEngine::Precision::FP32}};

INSTANTIATE_TEST_CASE_P(Gather_v7_big_04,
                        Gather7LayerTest,
                        ::testing::Combine(::testing::Values(big_04_params_v7.params_shape_),
                                           ::testing::Values(big_04_params_v7.indices_shape_),
                                           ::testing::Values(std::make_tuple(big_04_params_v7.axis_,
                                                                             big_04_params_v7.batch_dims_)),
                                           ::testing::ValuesIn(big_04_params_v7.net_precisions_),
                                           ::testing::Values(big_04_params_v7.input_precision_),
                                           ::testing::Values(big_04_params_v7.output_precision_),
                                           ::testing::Values(big_04_params_v7.input_layout_),
                                           ::testing::Values(big_04_params_v7.output_layout_),
                                           ::testing::Values(big_04_params_v7.device_)),
                        Gather7LayerTest::getTestCaseName);

using ParamsVec = std::vector<std::reference_wrapper<const GatherTestParams>>;

const ParamsVec all_params_v1 = {smoke_01_params_v1_v7,
                                 smoke_02_params_v1_v7,
                                 smoke_03_params_v1_v7,
                                 smoke_04_params_v1_v7,
                                 smoke_07_ov_params_v1_v7,
                                 tacotron2_enc_params_v1_v7,
                                 lpcnet_enc_params_v1_v7,
                                 lpcnet_dec_params_v1_v7,
                                 ov_params_v1,
                                 big_01_params_v1_v7,
                                 big_02_params_v1_v7};

const ParamsVec all_params_v7 = {smoke_01_params_v1_v7,
                                 smoke_02_params_v1_v7,
                                 smoke_03_params_v1_v7,
                                 smoke_04_params_v1_v7,
                                 smoke_05_params_v7,
                                 smoke_06_params_v7,
                                 smoke_07_ov_params_v1_v7,
                                 smoke_08_ov_params_v7,
                                 smoke_09_ov_params_v7,
                                 smoke_10_ov_params_v7,
                                 smoke_11_ov_params_v7,
                                 tacotron2_enc_params_v1_v7,
                                 lpcnet_enc_params_v1_v7,
                                 lpcnet_dec_params_v1_v7,
                                 ov_params_v7,
                                 big_01_params_v1_v7,
                                 big_02_params_v1_v7,
                                 big_03_params_v7,
                                 big_04_params_v7};

template <typename ElementType, typename IndicesType>
void test_one_shape(const GatherTestParams& params, bool is_v7) {
    using devptr_t = CUDA::DevicePointer<void*>;
    using cdevptr_t = CUDA::DevicePointer<const void*>;
    using microseconds = std::chrono::duration<double, std::micro>;
    using milliseconds = std::chrono::duration<double, std::milli>;

    constexpr int NUM_ATTEMPTS = 20;
    constexpr milliseconds WARMUP_TIME{2000.0};

    CUDAPlugin::ThreadContext threadContext{{}};
    int out_size = 0;
    CUDAPlugin::OperationBase::Ptr operation = [&] {
        const bool optimizeOption = false;
        auto dict_param = std::make_shared<ngraph::op::v0::Parameter>(ngraph::element::from<ElementType>(),
                                                                      ngraph::PartialShape{params.params_shape_});
        auto indices_param = std::make_shared<ngraph::op::v0::Parameter>(ngraph::element::from<IndicesType>(),
                                                                         ngraph::PartialShape{params.indices_shape_});
        auto axis_constant = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::i64, ngraph::Shape({}), std::vector<int64_t>{params.axis_});
        auto node = is_v7 ? std::static_pointer_cast<ngraph::Node>(std::make_shared<ngraph::op::v7::Gather>(
                                dict_param->output(0), indices_param->output(0), axis_constant, params.batch_dims_))
                          : std::static_pointer_cast<ngraph::Node>(std::make_shared<ngraph::op::v1::Gather>(
                                dict_param->output(0), indices_param->output(0), axis_constant));

        out_size = ngraph::shape_size(node->get_output_shape(0));
        auto& registry = CUDAPlugin::OperationRegistry::getInstance();
        auto op = registry.createOperation(CUDAPlugin::CreationContext{threadContext.device(), optimizeOption},
                                           node,
                                           std::array{CUDAPlugin::TensorID{0}},
                                           std::array{CUDAPlugin::TensorID{0}});
        return op;
    }();
    const int dict_size = ngraph::shape_size(params.params_shape_);
    const int indices_size = ngraph::shape_size(params.indices_shape_);
    const auto dict_size_bytes = dict_size * sizeof(ElementType);
    const auto indices_size_bytes = indices_size * sizeof(IndicesType);
    const auto out_size_bytes = out_size * sizeof(ElementType);
    CUDA::Allocation dict_alloc = threadContext.stream().malloc(dict_size_bytes);
    CUDA::Allocation indices_alloc = threadContext.stream().malloc(indices_size_bytes);
    CUDA::Allocation axis_alloc = threadContext.stream().malloc(sizeof(int64_t));
    CUDA::Allocation out_alloc = threadContext.stream().malloc(out_size_bytes);
    std::vector<cdevptr_t> inputs{dict_alloc, indices_alloc, axis_alloc};
    std::vector<devptr_t> outputs{out_alloc};

    InferenceEngine::BlobMap empty;
    CUDAPlugin::CancellationToken token{};
    CUDAPlugin::CudaGraph graph{CUDAPlugin::CreationContext{CUDA::Device{}, false}, {}};
    CUDAPlugin::Profiler profiler{false, graph};
    CUDAPlugin::InferenceRequestContext context{empty, empty, threadContext, token, profiler};
    std::vector<IndicesType> indices = generate_indices<IndicesType>(params);
    std::vector<ElementType> dict(dict_size);
    std::random_device r_device;
    std::mt19937 mersenne_engine{r_device()};
    std::uniform_int_distribution<int> dist{std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
    auto gen_dict = [&dist, &mersenne_engine]() {
        return static_cast<IndicesType>(10.f * dist(mersenne_engine) / std::numeric_limits<int>::max());
    };
    std::generate(dict.begin(), dict.end(), gen_dict);

    auto& stream = context.getThreadContext().stream();
    stream.upload(dict_alloc, dict.data(), dict_size_bytes);
    stream.upload(indices_alloc, indices.data(), indices_size_bytes);
    CUDAPlugin::Workbuffers workbuffers{};

    // Warmup
    auto warm_cur = std::chrono::steady_clock::now();
    const auto warm_end = warm_cur + WARMUP_TIME;
    while (warm_cur <= warm_end) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
        warm_cur = std::chrono::steady_clock::now();
    }

    // Benchmark
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_ATTEMPTS; ++i) {
        operation->Execute(context, inputs, outputs, workbuffers);
        stream.synchronize();
    }
    const auto end = std::chrono::steady_clock::now();
    microseconds average_exec_time = (end - start) / NUM_ATTEMPTS;
    std::cout << std::fixed << std::setfill('0') << "Gather_v" << (is_v7 ? "7 " : "1 ")
              << CommonTestUtils::vec2str(params.params_shape_) << ", "
              << CommonTestUtils::vec2str(params.indices_shape_) << ", axis = " << params.axis_
              << ", batch_dims = " << params.batch_dims_ << ": " << average_exec_time.count() << " us\n";
}

template <typename ElementType, typename IndicesType>
void test_all_shapes(const ParamsVec& all_params, bool is_v7) {
    for (const auto& p : all_params) {
        test_one_shape<ElementType, IndicesType>(p, is_v7);
    }
}

struct Gather_v1_Benchmark : testing::Test {};

TEST_F(Gather_v1_Benchmark, DISABLED_benchmark) {
    const bool is_v7 = false;

    std::cout << "---Dicts: float, Indices: int32_t:\n";
    test_all_shapes<float, int32_t>(all_params_v1, is_v7);

    std::cout << "---Dicts: float, Indices: int64_t:\n";
    test_all_shapes<float, int64_t>(all_params_v1, is_v7);

    std::cout << "---Dicts: ngraph::float16, Indices: int32_t:\n";
    test_all_shapes<ngraph::float16, int32_t>(all_params_v1, is_v7);
}

struct Gather_v7_Benchmark : testing::Test {};

TEST_F(Gather_v7_Benchmark, DISABLED_benchmark) {
    const bool is_v7 = true;
    ;

    std::cout << "---Dicts: float, Indices: int32_t:\n";
    test_all_shapes<float, int32_t>(all_params_v7, is_v7);

    std::cout << "---Dicts: float, Indices: int64_t:\n";
    test_all_shapes<float, int64_t>(all_params_v7, is_v7);

    std::cout << "---Dicts: ngraph::float16, Indices: int32_t:\n";
    test_all_shapes<ngraph::float16, int32_t>(all_params_v7, is_v7);
}

}  // namespace
