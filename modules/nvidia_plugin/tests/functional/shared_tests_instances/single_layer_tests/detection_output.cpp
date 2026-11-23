// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/detection_output.hpp"

#include <cmath>
#include <vector>

#include "cuda_test_constants.hpp"

namespace {
using ov::test::DetectionOutputLayerTest;
using ov::test::ParamsWhichSizeDepends;

/**
 * DetectionOutput produces a list of detections [image_id, label, conf, xmin, ymin, xmax, ymax]
 * terminated by image_id == -1. The CUDA implementation runs NMS in parallel, so detections with
 * equal confidence may come out in a different order than in the reference implementation.
 * Hence, the comparison is order-insensitive: every reference detection must have a matching
 * detection in the plugin output.
 */
class CudaDetectionOutputLayerTest : public DetectionOutputLayerTest {
protected:
    static constexpr size_t kNumDataInDetectionBox = 7;
    static constexpr float kThreshold = 0.1f;

    using Box = std::array<float, kNumDataInDetectionBox>;

    static bool similar(float res, float ref) {
        const float abs_diff = std::fabs(res - ref);
        if (abs_diff <= kThreshold) {
            return true;
        }
        const float max = std::max(std::fabs(res), std::fabs(ref));
        return max != 0.0f && !std::isnan(res) && !std::isnan(ref) && abs_diff / max <= kThreshold;
    }

    static bool similar(const Box& res, const Box& ref) {
        for (size_t i = 0; i < kNumDataInDetectionBox; ++i) {
            if (!similar(res[i], ref[i])) {
                return false;
            }
        }
        return true;
    }

    static std::vector<Box> extract_boxes(const ov::Tensor& tensor) {
        OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32, "Unexpected DetectionOutput element type");
        OPENVINO_ASSERT(tensor.get_size() % kNumDataInDetectionBox == 0,
                        "DetectionOutput result should be divisible by ",
                        kNumDataInDetectionBox);
        const float* data = tensor.data<const float>();
        std::vector<Box> boxes;
        for (size_t i = 0; i < tensor.get_size(); i += kNumDataInDetectionBox) {
            if (data[i] == -1.0f) {
                break;
            }
            Box box;
            std::copy_n(data + i, kNumDataInDetectionBox, box.begin());
            boxes.push_back(box);
        }
        return boxes;
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        for (size_t out = 0; out < expected.size(); ++out) {
            ASSERT_EQ(expected[out].get_size(), actual[out].get_size());
            const auto ref_boxes = extract_boxes(expected[out]);
            auto res_boxes = extract_boxes(actual[out]);
            ASSERT_EQ(ref_boxes.size(), res_boxes.size()) << "Number of detections differs";

            std::vector<size_t> not_matched;
            for (size_t i = 0; i < ref_boxes.size(); ++i) {
                auto it = std::find_if(res_boxes.begin(), res_boxes.end(), [&](const Box& res) {
                    return similar(res, ref_boxes[i]);
                });
                if (it == res_boxes.end()) {
                    not_matched.push_back(i);
                    continue;
                }
                res_boxes.erase(it);
            }
            ASSERT_LE(not_matched.size(), ref_boxes.size() / 2)
                << "Too many reference detections are not found in the plugin output";
            // Boxes that differ only within NMS tie-breaking must at least have an identical confidence
            for (const auto i : not_matched) {
                auto it = std::find_if(res_boxes.begin(), res_boxes.end(), [&](const Box& res) {
                    return res[2] == ref_boxes[i][2];
                });
                ASSERT_NE(it, res_boxes.end()) << "Cannot find reference detection with index " << i;
                res_boxes.erase(it);
            }
        }
    }
};

TEST_P(CudaDetectionOutputLayerTest, Inference) { run(); }

/* =============== Detection Output =============== */

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

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput3In,
                         CudaDetectionOutputLayerTest,
                         ::testing::Combine(commonAttributes,
                                            ::testing::ValuesIn(specificParams3In),
                                            ::testing::ValuesIn(numberBatch),
                                            ::testing::Values(0.0f),
                                            ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
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

INSTANTIATE_TEST_SUITE_P(smoke_DetectionOutput5In,
                         CudaDetectionOutputLayerTest,
                         ::testing::Combine(commonAttributes,
                                            ::testing::ValuesIn(specificParams5In),
                                            ::testing::ValuesIn(numberBatch),
                                            ::testing::Values(objectnessScore),
                                            ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                         CudaDetectionOutputLayerTest::getTestCaseName);

/* =============== SSD-MobileNet case =============== */

const auto commonAttributesSSDMobileNet = ::testing::Combine(::testing::Values(91),    // numClasses
                                                             ::testing::Values(0),     // backgroundLabelId
                                                             ::testing::Values(100),   // topK
                                                             ::testing::Values(std::vector<int>{100}),  // keepTopK
                                                             ::testing::Values("caffe.PriorBoxParameter.CENTER_SIZE"),
                                                             ::testing::Values(0.60000002384185791f),  // nmsThreshold
                                                             ::testing::Values(0.30000001192092896f),  // confThreshold
                                                             ::testing::Values(true),    // clipAfterNms
                                                             ::testing::Values(false),   // clipBeforeNms
                                                             ::testing::Values(false));  // decreaseLabelId

const std::vector<ParamsWhichSizeDepends> specificParamsSSDMobileNetIn = {
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 7668}, {1, 174447}, {1, 2, 7668}, {}, {}},
};

INSTANTIATE_TEST_SUITE_P(DetectionOutputSSDMobileNetIn,
                         CudaDetectionOutputLayerTest,
                         ::testing::Combine(commonAttributesSSDMobileNet,
                                            ::testing::ValuesIn(specificParamsSSDMobileNetIn),
                                            ::testing::Values(1, 2, 4),
                                            ::testing::Values(0.0f),
                                            ::testing::Values(ov::test::utils::DEVICE_NVIDIA)),
                         CudaDetectionOutputLayerTest::getTestCaseName);

}  // namespace
