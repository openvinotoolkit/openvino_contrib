// Copyright (C) 2021-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernels/detection_output.hpp>
#include <shared_test_classes/single_layer/detection_output.hpp>
#include <vector>

#include "cuda_test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

template <typename TDataType>
struct TestDetectionOutputResult {
    CUDAPlugin::kernel::DetectionOutputResult<TDataType> data;

    static float threshold;

    template <typename TOtherDataType>
    static bool similar(const TDataType &res, const TOtherDataType &ref) {
        const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
        if (absoluteDifference <= threshold) {
            return true;
        }
        double max;
        if (sizeof(TDataType) < sizeof(TOtherDataType)) {
            max = std::max(CommonTestUtils::ie_abs(TOtherDataType(res)), CommonTestUtils::ie_abs(ref));
        } else {
            max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(TDataType(ref)));
        }
        double diff = static_cast<float>(absoluteDifference) / max;
        if (max == 0 || (diff > static_cast<float>(threshold)) || std::isnan(static_cast<float>(res)) ||
            std::isnan(static_cast<float>(ref))) {
            return false;
        }
        return true;
    }

    template <typename TOtherDataType>
    bool operator==(const TestDetectionOutputResult<TOtherDataType> &other) const {
        return similar(this->data.img, other.data.img) && similar(this->data.cls, other.data.cls) &&
               similar(this->data.conf, other.data.conf) && similar(this->data.xmin, other.data.xmin) &&
               similar(this->data.ymin, other.data.ymin) && similar(this->data.xmax, other.data.xmax) &&
               similar(this->data.ymax, other.data.ymax);
    }

    template <typename TOtherDataType>
    bool operator!=(const TestDetectionOutputResult<TOtherDataType> &other) const {
        return !(this->data.operator==(other));
    }
};

template <typename TDataType>
float TestDetectionOutputResult<TDataType>::threshold = 0.0;

class CudaDetectionOutputLayerTest : public DetectionOutputLayerTest {
private:
    static constexpr size_t kNumDataInDetectionBox = 7;

    template <class T_IE, class T_NGRAPH>
    static void Compare(const T_NGRAPH *expected, const T_IE *actual, std::size_t size, float threshold) {
        assert(size % kNumDataInDetectionBox == 0 && "Results should be dividable by kNumDataInDetectionBox");
        const size_t num_results = size / kNumDataInDetectionBox;
        TestDetectionOutputResult<T_NGRAPH>::threshold = threshold;
        TestDetectionOutputResult<T_IE>::threshold = threshold;
        std::vector<TestDetectionOutputResult<T_NGRAPH>> ngraph_results(num_results);
        std::vector<TestDetectionOutputResult<T_IE>> ie_results(num_results);
        for (std::size_t i = 0, res = 0; i < size; i += kNumDataInDetectionBox, ++res) {
            auto &ngraph_result = ngraph_results[res];
            ngraph_result.data.img = expected[i + 0];
            ngraph_result.data.cls = expected[i + 1];
            ngraph_result.data.conf = expected[i + 2];
            ngraph_result.data.xmin = expected[i + 3];
            ngraph_result.data.ymin = expected[i + 4];
            ngraph_result.data.xmax = expected[i + 5];
            ngraph_result.data.ymax = expected[i + 6];
            auto &ie_result = ie_results[res];
            ie_result.data.img = actual[i + 0];
            ie_result.data.cls = actual[i + 1];
            ie_result.data.conf = actual[i + 2];
            ie_result.data.xmin = actual[i + 3];
            ie_result.data.ymin = actual[i + 4];
            ie_result.data.xmax = actual[i + 5];
            ie_result.data.ymax = actual[i + 6];
        }
        std::vector<std::pair<std::size_t, TestDetectionOutputResult<T_NGRAPH>>> not_matched_ngraph_results;
        for (std::size_t i = 0; i < num_results; ++i) {
            const auto &ref = ngraph_results[i];
            auto res = std::find(ie_results.cbegin(), ie_results.cend(), ref);
            if (res == ie_results.end()) {
                not_matched_ngraph_results.push_back(std::make_pair(i, ref));
                continue;
            }
            ie_results.erase(res);
        }
        const auto precent = static_cast<float>(not_matched_ngraph_results.size()) / ngraph_results.size();
        if (precent > 0.5) {
            IE_THROW() << "Too many elements not found in reference implementation "
                       << "with relative comparison of values with threshold " << threshold;
        }
        for (const auto &[i, ref] : not_matched_ngraph_results) {
            auto res = std::find_if(ie_results.begin(), ie_results.end(), [ref = ref](const auto &res) {
                return ref.data.conf == res.data.conf;
            });
            if (res == ie_results.end()) {
                IE_THROW() << "Cannot find object (index=" << i
                           << ") with relative comparison of values with threshold " << threshold << " failed";
            }
            ie_results.erase(res);
        }
    }

public:
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs) override {
        for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
            const auto &expected = expectedOutputs[outputIndex].second;
            const auto &actual = actualOutputs[outputIndex];

            ASSERT_EQ(expected.size(), actual->byteSize());

            size_t expSize = 0;
            size_t actSize = 0;

            const auto &expectedBuffer = expected.data();
            auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(actual);
            IE_ASSERT(memory);
            const auto lockedMemory = memory->wmap();
            const auto actualBuffer = lockedMemory.as<const std::uint8_t *>();

            const float *expBuf = reinterpret_cast<const float *>(expectedBuffer);
            const float *actBuf = reinterpret_cast<const float *>(actualBuffer);
            for (size_t i = 0; i < actual->size(); i += kNumDataInDetectionBox) {
                if (expBuf[i] == -1) break;
                expSize += kNumDataInDetectionBox;
            }
            for (size_t i = 0; i < actual->size(); i += kNumDataInDetectionBox) {
                if (actBuf[i] == -1) break;
                actSize += kNumDataInDetectionBox;
            }
            ASSERT_EQ(expSize, actSize);
            Compare<float>(expBuf, actBuf, expSize, 0.1);
        }
    }
};

TEST_P(CudaDetectionOutputLayerTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto params = GetParam();

    Run();
}

/* =============== Detection Output =============== */

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::BF16,
};

const int numClasses = 11;
const int backgroundLabelId = 12;
const std::vector<int> topK = {75};
const std::vector<std::vector<int>> keepTopK = {{50}, {100}};
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CORNER", "caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThreshold = 0.5f;
const float confidenceThreshold = 0.3f;
const std::vector<bool> clipAfterNms = {true, false};
const std::vector<bool> clipBeforeNms = {true, false};
const std::vector<bool> decreaseLabelId = {true, false};
const float objectnessScore = 0.4f;
const std::vector<size_t> numberBatch = {1, 2, 8};

const auto commonAttributes = ::testing::Combine(::testing::Values(numClasses),
                                                 ::testing::Values(backgroundLabelId),
                                                 ::testing::ValuesIn(topK),
                                                 ::testing::ValuesIn(keepTopK),
                                                 ::testing::ValuesIn(codeType),
                                                 ::testing::Values(nmsThreshold),
                                                 ::testing::Values(confidenceThreshold),
                                                 ::testing::ValuesIn(clipAfterNms),
                                                 ::testing::ValuesIn(clipBeforeNms),
                                                 ::testing::ValuesIn(decreaseLabelId));

/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams3In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {}, {}},
    ParamsWhichSizeDepends{false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {}, {}},

    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {}, {}},
    ParamsWhichSizeDepends{false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {}, {}}};

const auto params3Inputs = ::testing::Combine(commonAttributes,
                                              ::testing::ValuesIn(specificParams3In),
                                              ::testing::ValuesIn(numberBatch),
                                              ::testing::Values(0.0f),
                                              ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_DetectionOutput3In,
                        CudaDetectionOutputLayerTest,
                        params3Inputs,
                        CudaDetectionOutputLayerTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams5In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, false, true, 1, 1, {1, 660}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 660}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, false, true, 1, 1, {1, 660}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 660}},

    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, false, false, 10, 10, {1, 660}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 660}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, false, false, 10, 10, {1, 660}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 660}}};

const auto params5Inputs = ::testing::Combine(commonAttributes,
                                              ::testing::ValuesIn(specificParams5In),
                                              ::testing::ValuesIn(numberBatch),
                                              ::testing::Values(objectnessScore),
                                              ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(smoke_DetectionOutput5In,
                        CudaDetectionOutputLayerTest,
                        params5Inputs,
                        CudaDetectionOutputLayerTest::getTestCaseName);

/* =============== SSD-MobileNet case =============== */
const int numClassesSSDMobileNet = 91;
const int backgroundLabelIdSSDMobileNet = 0;
const std::vector<int> topKSSDMobileNet = {100};
const std::vector<std::vector<int>> keepTopKSSDMobileNet = {{100}};
const std::vector<std::string> codeTypeSSDMobileNet = {"caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThresholdSSDMobileNet = 0.60000002384185791f;
const float confidenceThresholdSSDMobileNet = 0.30000001192092896f;
const std::vector<bool> clipAfterNmsSSDMobileNet = {true};
const std::vector<bool> clipBeforeNmsSSDMobileNet = {false};
const std::vector<bool> decreaseLabelIdSSDMobileNet = {false};
const float objectnessScoreSSDMobileNet = 0.0f;
const std::vector<size_t> numberBatchSSDMobileNet = {1, 2, 4};

const auto commonAttributesSSDMobileNet = ::testing::Combine(::testing::Values(numClassesSSDMobileNet),
                                                             ::testing::Values(backgroundLabelIdSSDMobileNet),
                                                             ::testing::ValuesIn(topKSSDMobileNet),
                                                             ::testing::ValuesIn(keepTopKSSDMobileNet),
                                                             ::testing::ValuesIn(codeTypeSSDMobileNet),
                                                             ::testing::Values(nmsThresholdSSDMobileNet),
                                                             ::testing::Values(confidenceThresholdSSDMobileNet),
                                                             ::testing::ValuesIn(clipAfterNmsSSDMobileNet),
                                                             ::testing::ValuesIn(clipBeforeNmsSSDMobileNet),
                                                             ::testing::ValuesIn(decreaseLabelIdSSDMobileNet));

const std::vector<ParamsWhichSizeDepends> specificParamsSSDMobileNetIn = {
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 7668}, {1, 174447}, {1, 2, 7668}, {}, {}},
};

const auto paramsSSDMobileNetInputs = ::testing::Combine(commonAttributesSSDMobileNet,
                                                         ::testing::ValuesIn(specificParamsSSDMobileNetIn),
                                                         ::testing::ValuesIn(numberBatchSSDMobileNet),
                                                         ::testing::Values(objectnessScoreSSDMobileNet),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(DetectionOutputSSDMobileNetIn,
                        CudaDetectionOutputLayerTest,
                        paramsSSDMobileNetInputs,
                        CudaDetectionOutputLayerTest::getTestCaseName);

///* =============== EfficientDet case =============== */
const int numClassesEfficientDet = 90;
const int backgroundLabelIdEfficientDet = 91;
const std::vector<int> topKEfficientDet = {100};
const std::vector<std::vector<int>> keepTopKEfficientDet = {{100}};
const std::vector<std::string> codeTypeEfficientDet = {"caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThresholdEfficientDet = 0.60000002384185791f;
const float confidenceThresholdEfficientDet = 0.20000000298023224f;
const std::vector<bool> clipAfterNmsEfficientDet = {false};
const std::vector<bool> clipBeforeNmsEfficientDet = {false};
const std::vector<bool> decreaseLabelIdEfficientDet = {false};
const float objectnessScoreEfficientDet = 0.0f;
const std::vector<size_t> numberBatchEfficientDet = {1};

const auto commonAttributesEfficientDet = ::testing::Combine(::testing::Values(numClassesEfficientDet),
                                                             ::testing::Values(backgroundLabelIdEfficientDet),
                                                             ::testing::ValuesIn(topKEfficientDet),
                                                             ::testing::ValuesIn(keepTopKEfficientDet),
                                                             ::testing::ValuesIn(codeTypeEfficientDet),
                                                             ::testing::Values(nmsThresholdEfficientDet),
                                                             ::testing::Values(confidenceThresholdEfficientDet),
                                                             ::testing::ValuesIn(clipAfterNmsEfficientDet),
                                                             ::testing::ValuesIn(clipBeforeNmsEfficientDet),
                                                             ::testing::ValuesIn(decreaseLabelIdEfficientDet));

const std::vector<ParamsWhichSizeDepends> specificParamsEfficientDetIn = {
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 306900}, {1, 6905250}, {1, 2, 306900}, {}, {}},
};

const auto paramsEfficientDetInputs = ::testing::Combine(commonAttributesEfficientDet,
                                                         ::testing::ValuesIn(specificParamsEfficientDetIn),
                                                         ::testing::ValuesIn(numberBatchEfficientDet),
                                                         ::testing::Values(objectnessScoreEfficientDet),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CUDA));

// NOTE: Too many elements with similar confidence that leads test to fail
INSTANTIATE_TEST_CASE_P(DISABLED_DetectionOutputEfficientDetIn,
                        CudaDetectionOutputLayerTest,
                        paramsEfficientDetInputs,
                        CudaDetectionOutputLayerTest::getTestCaseName);

// ------------- Benchmark -------------
#include "benchmark.hpp"

namespace LayerTestsDefinitions {
namespace benchmark {

struct CudaDetectionOutputLayerBenchmarkTest : BenchmarkLayerTest<CudaDetectionOutputLayerTest> {};

TEST_P(CudaDetectionOutputLayerBenchmarkTest, DISABLED_benchmark) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run("DetectionOutput", std::chrono::milliseconds(2000), 5);
}

const auto paramsEfficientDetInputs = ::testing::Combine(commonAttributesEfficientDet,
                                                         ::testing::ValuesIn(specificParamsEfficientDetIn),
                                                         ::testing::ValuesIn(numberBatchEfficientDet),
                                                         ::testing::Values(objectnessScoreEfficientDet),
                                                         ::testing::Values(CommonTestUtils::DEVICE_CUDA));

INSTANTIATE_TEST_CASE_P(DetectionOutputEfficientDetIn,
                        CudaDetectionOutputLayerBenchmarkTest,
                        paramsEfficientDetInputs,
                        CudaDetectionOutputLayerBenchmarkTest::getTestCaseName);

}  // namespace benchmark
}  // namespace LayerTestsDefinitions

}  // namespace
