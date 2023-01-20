// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/mat_mul.hpp"

#include <cuda_test_constants.hpp>
#include <vector>

#include "finite_comparer.hpp"

namespace LayerTestsDefinitions {

class MatMulLayerTest : public FiniteComparer<MatMulTest> {
protected:
    void SetUp() override {
        MatMulTest::SetUp();

        auto params = this->GetParam();
        auto netPrecision = std::get<1>(params);
        if (netPrecision.getPrecVal() == InferenceEngine::Precision::FP16) {
            this->infinity_value = std::numeric_limits<std::uint16_t>::max();
        }
    }
};

TEST_P(MatMulLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto params = GetParam();
    inPrc = std::get<2>(params);
    outPrc = std::get<3>(params);

    Run();
}

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32,
};

// General
const std::vector<ShapeRelatedParams> smokeShapeRelatedParams = {
    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    {{{3, 2, 10, 10}, false}, {{3, 2, 10, 20}, false}},
    {{{3, 2, 10, 10}, true}, {{3, 2, 10, 20}, false}},
    {{{3, 2, 10, 20}, false}, {{3, 2, 10, 20}, true}},
    {{{3, 2, 20, 10}, true}, {{3, 2, 10, 20}, true}},

    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    {{{2, 10, 10}, false}, {{2, 10, 20}, false}},
    {{{2, 10, 10}, true}, {{2, 10, 20}, false}},
    {{{2, 10, 20}, false}, {{2, 10, 20}, true}},
    {{{2, 20, 10}, true}, {{2, 10, 20}, true}},

    // ND x ND: [B, ..., X, Y] x [B, ..., Y, Z] => [B, ..., X, Z]
    {{{10, 10}, false}, {{10, 20}, false}},
    {{{10, 10}, true}, {{10, 20}, false}},
    {{{10, 20}, false}, {{10, 20}, true}},
    {{{20, 10}, true}, {{10, 20}, true}},

    // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
    {{{2, 10, 10}, false}, {{10}, false}},
    {{{2, 10, 10}, true}, {{10}, false}},
    {{{2, 10, 20}, false}, {{20}, true}},
    {{{2, 20, 10}, true}, {{20}, true}},

    // ND x 1D: [B, ..., X, Y] x [Y] -> [B, ..., X, Y] x [Y, 1] -> [B, ..., X, 1] => [B, ..., X]
    {{{10, 10}, false}, {{10}, false}},
    {{{10, 10}, true}, {{10}, false}},
    {{{10, 20}, false}, {{20}, true}},
    {{{20, 10}, true}, {{20}, true}},

    // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
    {{{10}, false}, {{10, 20}, false}},
    {{{10}, true}, {{10, 20}, false}},
    {{{20}, false}, {{10, 20}, true}},
    {{{20}, true}, {{10, 20}, true}},

    // 1D x ND: [X] x [B, ..., X, Y] -> [1, X] x [B, ..., X, Y] -> [B, ..., 1, Y] => [B, ..., Y]
    {{{10}, false}, {{2, 10, 20}, false}},
    {{{10}, true}, {{2, 10, 20}, false}},
    {{{20}, false}, {{2, 10, 20}, true}},
    {{{20}, true}, {{2, 10, 20}, true}},

    // 1D x 1D: [X] x [X] -> [1, X] x [X, 1] -> [1, 1] => [] (scalar)
    {{{10}, false}, {{10}, false}},
    {{{10}, true}, {{10}, false}},
    {{{10}, false}, {{10}, true}},
    {{{10}, true}, {{10}, true}},
};

// NOTE: Resnet-50 shapes
const std::vector<ShapeRelatedParams> resnet50ShapeRelatedParams = {
    {{{1, 2048}, false}, {{2048, 1001}, false}},
    {{{1, 2048}, false}, {{1001, 2048}, true}},
};

// NOTE: VGG-16 shapes
const std::vector<ShapeRelatedParams> vgg16ShapeRelatedParams = {
    {{{1, 25088}, false}, {{4096, 25088}, true}},
    {{{1, 25088}, false}, {{25088, 4096}, false}},
    {{{1, 4096}, false}, {{4096, 4096}, true}},
    {{{1, 4096}, false}, {{4096, 4096}, false}},
    {{{1, 4096}, false}, {{1000, 4096}, true}},
    {{{1, 4096}, false}, {{4096, 1000}, false}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_CASE_P(smoke_MatMul,
                        MatMulLayerTest,
                        ::testing::Combine(::testing::ValuesIn(smokeShapeRelatedParams),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(secondaryInputTypes),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_Resnet50,
                        MatMulLayerTest,
                        ::testing::Combine(::testing::ValuesIn(resnet50ShapeRelatedParams),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(secondaryInputTypes),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(MatMul_VGG16,
                        MatMulLayerTest,
                        ::testing::Combine(::testing::ValuesIn(vgg16ShapeRelatedParams),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(secondaryInputTypes),
                                           ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
                                           ::testing::Values(additional_config)),
                        MatMulLayerTest::getTestCaseName);

// =============================================================================
// clang-format off
// {AUTOGENERATED_TESTS_BEGIN_TAG}

// Attrs:  {'transpose_a': 'false', 'transpose_b': 'false'}
// In:     (1, 1, 1000), (1, 1000, 512)
// Out:    (1, 1, 512)
// Operators: 'Tacotron2-decoder_iter:opid129' [FP32], 'Tacotron2-graph-transform-cuda-decoder_iter:opid69' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid129,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1, 1000}, false }, { {1, 1000, 512}, false } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1, 1024), (128, 1024)
// Out:    (1, 1, 128)
// Operators: 'Tacotron2-decoder_iter:opid98' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid98,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1, 1024}, false }, { {128, 1024}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1000, 128), (1, 128)
// Out:    (1, 1000, 1)
// Operators: 'Tacotron2-decoder_iter:opid117' [FP32], 'Tacotron2-graph-transform-cuda-decoder_iter:opid62' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid117,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1000, 128}, false }, { {1, 128}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1000, 32), (128, 32)
// Out:    (1, 1000, 128)
// Operators: 'Tacotron2-decoder_iter:opid111' [FP32], 'Tacotron2-graph-transform-cuda-decoder_iter:opid57' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid111,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1000, 32}, false }, { {128, 32}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1000, 512), (128, 512)
// Out:    (1, 1000, 128)
// Operators: 'Tacotron2-encoder:opid79' [FP32], 'Tacotron2-graph-transform-cuda-encoder:opid56' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_encoder_opid79,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1000, 512}, false }, { {128, 512}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1536), (1, 1536)
// Out:    (1, 1)
// Operators: 'Tacotron2-decoder_iter:opid164' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid164,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1536}, false }, { {1, 1536}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1536), (1001, 1536)
// Out:    (1, 1001)
// Operators: 'googlenet-v4-tf:opid804' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_googlenet_v4_tf_opid804,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1536}, false }, { {1001, 1536}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 1536), (80, 1536)
// Out:    (1, 80)
// Operators: 'Tacotron2-decoder_iter:opid159' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid159,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 1536}, false }, { {80, 1536}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 2048), (1000, 2048)
// Out:    (1, 1000)
// Operators: 'resnet-50-caffe2:opid292' [FP16, FP32], 'resnet-50-pytorch:opid299' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_resnet_50_caffe2_opid292,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 2048}, false }, { {1000, 2048}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 2048), (1001, 2048)
// Out:    (1, 1001)
// Operators: 'resnet-50-tf:opid288' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_resnet_50_tf_opid288,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 2048}, false }, { {1001, 2048}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 25088), (4096, 25088)
// Out:    (1, 4096)
// Operators: 'vgg16-IR:opid81' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_vgg16_IR_opid81,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 25088}, false }, { {4096, 25088}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 256), (256, 256)
// Out:    (1, 256)
// Operators: 'Tacotron2-decoder_iter:opid35' [FP32], 'Tacotron2-graph-transform-cuda-decoder_iter:opid24' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid35,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 256}, false }, { {256, 256}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 4096), (1000, 4096)
// Out:    (1, 1000)
// Operators: 'vgg16-IR:opid105' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_vgg16_IR_opid105,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 4096}, false }, { {1000, 4096}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 4096), (4096, 4096)
// Out:    (1, 4096)
// Operators: 'vgg16-IR:opid93' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_vgg16_IR_opid93,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 4096}, false }, { {4096, 4096}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (1, 80), (256, 80)
// Out:    (1, 256)
// Operators: 'Tacotron2-decoder_iter:opid2' [FP32], 'Tacotron2-graph-transform-cuda-decoder_iter:opid14' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_Tacotron2_decoder_iter_opid2,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {1, 80}, false }, { {256, 80}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (100, 1024), (360, 1024)
// Out:    (100, 360)
// Operators: 'mask_rcnn_inception_v2_coco:opid424' [FP32], 'mask_rcnn_inception_v2_coco:opid425' [FP16]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_mask_rcnn_inception_v2_coco_opid424,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {100, 1024}, false }, { {360, 1024}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (100, 1024), (91, 1024)
// Out:    (100, 91)
// Operators: 'mask_rcnn_inception_v2_coco:opid441' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_mask_rcnn_inception_v2_coco_opid441,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {100, 1024}, false }, { {91, 1024}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (4096, 16), (512, 16)
// Out:    (4096, 512)
// Operators: 'LPCnet-lpcnet_dec:opid70' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_LPCnet_lpcnet_dec_opid70,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {4096, 16}, false }, { {512, 16}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (64, 1024), (6272, 1024)
// Out:    (64, 6272)
// Operators: 'GAN:opid35' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_GAN_opid35,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {64, 1024}, false }, { {6272, 1024}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (64, 62), (1024, 62)
// Out:    (64, 1024)
// Operators: 'GAN:opid2' [FP16, FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_GAN_opid2,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {64, 62}, false }, { {1024, 62}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);


// Attrs:  {'transpose_a': 'false', 'transpose_b': 'true'}
// In:     (64, 64, 128), (128, 128)
// Out:    (64, 64, 128)
// Operators: 'LPCnet-lpcnet_enc:opid24' [FP32], 'LPCnet-lpcnet_enc:opid29' [FP32]
INSTANTIATE_TEST_CASE_P(
    autogen_MatMul_LPCnet_lpcnet_enc_opid24,
    MatMulLayerTest,
    ::testing::Combine(
        ::testing::Values(ShapeRelatedParams{ { {64, 64, 128}, false }, { {128, 128}, true } }),
        ::testing::ValuesIn(std::vector<InferenceEngine::Precision>({InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16})),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(std::vector<ngraph::helpers::InputLayerType> {ngraph::helpers::InputLayerType::CONSTANT, ngraph::helpers::InputLayerType::PARAMETER}), // secondary input types
        ::testing::Values(CommonTestUtils::DEVICE_NVIDIA),
        ::testing::Values(std::map<std::string, std::string> {})), // additional config
    MatMulLayerTest::getTestCaseName);

// {AUTOGENERATED_TESTS_END_TAG}
// clang-format on
// =============================================================================

}  // namespace
}  // namespace LayerTestsDefinitions
