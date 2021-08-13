// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_test_constants.hpp>
#include "single_layer_tests/gather.hpp"
#include <vector>

namespace LayerTestsDefinitions {

// The class was inherited to change getTestCaseName() method
// to disable printing of the indices tensors because of their large size in the tests
// printing of them sometimes leads to seg fault
class CudaGatherLayerTest : public GatherLayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<gatherParamsTuple> &obj) {
        int axis;
        std::vector<int> indices;
        std::vector<size_t> indicesShape, inputShape;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        std::string targetName;
        std::tie(indices, indicesShape, axis, inputShape, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName) = obj.param;
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
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

struct GatherTestParams {
    std::vector<size_t> params_shape_;
    std::vector<size_t> indices_shape_;
    int axis_ = 0;
    std::vector<InferenceEngine::Precision> net_precisions_ = { InferenceEngine::Precision::FP32,
                                                                InferenceEngine::Precision::FP16 };
    InferenceEngine::Precision input_precision_ = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision output_precision_ = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Layout input_layout_ = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout output_layout_ = InferenceEngine::Layout::ANY;
    LayerTestsUtils::TargetDevice device_ = CommonTestUtils::DEVICE_CUDA;
};

std::vector<int> generate_indices(const GatherTestParams& test_params) {
    static std::random_device r_device;
    static std::default_random_engine r_engine { r_device() };

    const auto params_shape_size = test_params.params_shape_.size();
    const auto axis = test_params.axis_;
    const unsigned positive_axis = axis >= 0 ? axis : axis + params_shape_size;
    if (positive_axis >= params_shape_size) {
        THROW_IE_EXCEPTION << "positive_axis >= params_shape_size: " << positive_axis << " >= "
                           << params_shape_size;
    }
    std::uniform_int_distribution<int> distr(0, test_params.params_shape_[positive_axis] - 1);

    const auto indices_size = ngraph::shape_size(test_params.indices_shape_);
    std::vector<int> indices(indices_size);
    std::generate(indices.begin(), indices.end(), [&]() { return distr(r_engine); });
    return indices;
}


const GatherTestParams smoke_01_params = { { 4, 3 }, { 2 } };

INSTANTIATE_TEST_CASE_P(smoke_Gather_01, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(smoke_01_params)),
                            ::testing::Values(smoke_01_params.indices_shape_),
                            ::testing::Values(smoke_01_params.axis_),
                            ::testing::Values(smoke_01_params.params_shape_),
                            ::testing::ValuesIn(smoke_01_params.net_precisions_),
                            ::testing::Values(smoke_01_params.input_precision_),
                            ::testing::Values(smoke_01_params.output_precision_),
                            ::testing::Values(smoke_01_params.input_layout_),
                            ::testing::Values(smoke_01_params.output_layout_),
                            ::testing::Values(smoke_01_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams smoke_02_params = { { 1, 2, 3 }, { 1 }, 1 };

INSTANTIATE_TEST_CASE_P(smoke_Gather_02, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(smoke_02_params)),
                            ::testing::Values(smoke_02_params.indices_shape_),
                            ::testing::Values(smoke_02_params.axis_),
                            ::testing::Values(smoke_02_params.params_shape_),
                            ::testing::ValuesIn(smoke_02_params.net_precisions_),
                            ::testing::Values(smoke_02_params.input_precision_),
                            ::testing::Values(smoke_02_params.output_precision_),
                            ::testing::Values(smoke_02_params.input_layout_),
                            ::testing::Values(smoke_02_params.output_layout_),
                            ::testing::Values(smoke_02_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams smoke_03_params = { { 1, 2, 3 }, { 7 }, -2 };

INSTANTIATE_TEST_CASE_P(smoke_Gather_03, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(smoke_03_params)),
                            ::testing::Values(smoke_03_params.indices_shape_),
                            ::testing::Values(smoke_03_params.axis_),
                            ::testing::Values(smoke_03_params.params_shape_),
                            ::testing::ValuesIn(smoke_03_params.net_precisions_),
                            ::testing::Values(smoke_03_params.input_precision_),
                            ::testing::Values(smoke_03_params.output_precision_),
                            ::testing::Values(smoke_03_params.input_layout_),
                            ::testing::Values(smoke_03_params.output_layout_),
                            ::testing::Values(smoke_03_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

const GatherTestParams smoke_04_params = { { 1, 2, 3 }, { 7, 5 }, 1 };

INSTANTIATE_TEST_CASE_P(smoke_Gather_04, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(smoke_04_params)),
                            ::testing::Values(smoke_04_params.indices_shape_),
                            ::testing::Values(smoke_04_params.axis_),
                            ::testing::Values(smoke_04_params.params_shape_),
                            ::testing::ValuesIn(smoke_04_params.net_precisions_),
                            ::testing::Values(smoke_04_params.input_precision_),
                            ::testing::Values(smoke_04_params.output_precision_),
                            ::testing::Values(smoke_04_params.input_layout_),
                            ::testing::Values(smoke_04_params.output_layout_),
                            ::testing::Values(smoke_04_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

// ------------- LPCNet shapes -------------
const GatherTestParams lpcnet_enc_params = { { 256, 64 }, { 64, 64, 1 } };

INSTANTIATE_TEST_CASE_P(Gather_LPCnet_shapes_enc, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(lpcnet_enc_params)),
                            ::testing::Values(lpcnet_enc_params.indices_shape_),
                            ::testing::Values(lpcnet_enc_params.axis_),
                            ::testing::Values(lpcnet_enc_params.params_shape_),
                            ::testing::ValuesIn(lpcnet_enc_params.net_precisions_),
                            ::testing::Values(lpcnet_enc_params.input_precision_),
                            ::testing::Values(lpcnet_enc_params.output_precision_),
                            ::testing::Values(lpcnet_enc_params.input_layout_),
                            ::testing::Values(lpcnet_enc_params.output_layout_),
                            ::testing::Values(lpcnet_enc_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);


const GatherTestParams lpcnet_dec_params = { { 256, 128 }, { 64, 64, 3 } };

INSTANTIATE_TEST_CASE_P(Gather_LPCnet_shapes_dec, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(lpcnet_dec_params)),
                            ::testing::Values(lpcnet_dec_params.indices_shape_),
                            ::testing::Values(lpcnet_dec_params.axis_),
                            ::testing::Values(lpcnet_dec_params.params_shape_),
                            ::testing::ValuesIn(lpcnet_dec_params.net_precisions_),
                            ::testing::Values(lpcnet_dec_params.input_precision_),
                            ::testing::Values(lpcnet_dec_params.output_precision_),
                            ::testing::Values(lpcnet_dec_params.input_layout_),
                            ::testing::Values(lpcnet_dec_params.output_layout_),
                            ::testing::Values(lpcnet_dec_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);


// ------------- Big shapes -------------
const GatherTestParams ov_v1_params = { { 6, 12, 10, 24 }, { 15, 4, 20, 28 }, 1 };

INSTANTIATE_TEST_CASE_P(Gather_OV_v1, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(ov_v1_params)),
                            ::testing::Values(ov_v1_params.indices_shape_),
                            ::testing::Values(ov_v1_params.axis_),
                            ::testing::Values(ov_v1_params.params_shape_),
                            ::testing::ValuesIn(ov_v1_params.net_precisions_),
                            ::testing::Values(ov_v1_params.input_precision_),
                            ::testing::Values(ov_v1_params.output_precision_),
                            ::testing::Values(ov_v1_params.input_layout_),
                            ::testing::Values(ov_v1_params.output_layout_),
                            ::testing::Values(ov_v1_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);


const GatherTestParams big_01_params = { { 2048, 32, 2048 }, { 2, 8, 1, 1 }, 0,
                                         { InferenceEngine::Precision::FP32 } };

INSTANTIATE_TEST_CASE_P(Gather_big_01, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(big_01_params)),
                            ::testing::Values(big_01_params.indices_shape_),
                            ::testing::Values(big_01_params.axis_),
                            ::testing::Values(big_01_params.params_shape_),
                            ::testing::ValuesIn(big_01_params.net_precisions_),
                            ::testing::Values(big_01_params.input_precision_),
                            ::testing::Values(big_01_params.output_precision_),
                            ::testing::Values(big_01_params.input_layout_),
                            ::testing::Values(big_01_params.output_layout_),
                            ::testing::Values(big_01_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);


const GatherTestParams big_02_params = { { 2048, 32, 2048 }, { 2, 8, 1, 1 }, 2,
                                         { InferenceEngine::Precision::FP16 } };

INSTANTIATE_TEST_CASE_P(Gather_big_02, CudaGatherLayerTest,
                        ::testing::Combine(
                            ::testing::Values(generate_indices(big_02_params)),
                            ::testing::Values(big_02_params.indices_shape_),
                            ::testing::Values(big_02_params.axis_),
                            ::testing::Values(big_02_params.params_shape_),
                            ::testing::ValuesIn(big_02_params.net_precisions_),
                            ::testing::Values(big_02_params.input_precision_),
                            ::testing::Values(big_02_params.output_precision_),
                            ::testing::Values(big_02_params.input_layout_),
                            ::testing::Values(big_02_params.output_layout_),
                            ::testing::Values(big_02_params.device_)),
                        CudaGatherLayerTest::getTestCaseName);

} // namespace
