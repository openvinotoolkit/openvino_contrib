// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <error.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace LayerTestsDefinitions {

namespace details {

template <typename T>
static T findAverage(const std::vector<std::uint8_t>& output) {
    const auto* ptr = static_cast<const T*>(static_cast<const void*>(output.data()));
    const auto size = output.size() / sizeof(T);
    const T absSum =
        std::accumulate(ptr, ptr + size, static_cast<T>(0), [](T a, T b) { return std::abs(a) + std::abs(b); });
    const T average = absSum / static_cast<T>(size);
    std::cout << "average absolute :" << average << '\n';
    return average;
}

struct BlobLimits {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    float avg = 0.0f;
    float absMin = std::numeric_limits<float>::max();
    float absMax = 0.0f;
    float absAvg = 0.0f;
};

template <typename T>
static BlobLimits findLimits(const std::vector<std::uint8_t>& output, BlobLimits& bl) {
    bl = BlobLimits{};
    const auto* ptr = static_cast<const T*>(static_cast<const void*>(output.data()));
    float sum = 0.0f;
    float absSum = 0.0f;
    size_t size = output.size() / sizeof(T);
    for (size_t i = 0; i < size; ++i) {
        const auto el = static_cast<float>(ptr[i]);
        const auto absEl = std::abs(el);
        bl.min = el < bl.min ? el : bl.min;
        bl.max = el > bl.max ? el : bl.max;
        bl.absMin = absEl < bl.absMin ? absEl : bl.absMin;
        bl.absMax = absEl > bl.absMax ? absEl : bl.absMax;
        sum += el;
        absSum += absEl;
    }
    bl.avg = sum / size;
    bl.absAvg = absSum / size;

    std::cout << "min = " << bl.min << ", max = " << bl.max << ", avg = " << bl.avg << '\n';
    std::cout << "absMin = " << bl.absMin << ", absMax = " << bl.absMax << ", absAvg = " << bl.absAvg << '\n';

    return bl;
}

}  // namespace details

/**
 * @brief This class is the base class for AverageFinder class.
 * It is used to set the threshold for LayerTestsUtils::LayerTestsCommon::Comapare() functions
 * accordingly to the average absolute value of the reference output of a single layer test class.
 * To use it, threshold_base should be set in the derived class.
 * threshold = average * threshold_base
 * For now can be used only for the operations with one output.
 */
class AverageFinderBase : virtual public LayerTestsUtils::LayerTestsCommon {
    virtual std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override {
        using namespace details;
        const auto refOutputs = LayerTestsCommon::CalculateRefs();
        if (refOutputs.size() == 1) {
            const auto& type = refOutputs[0].first;
            const auto& output = refOutputs[0].second;
            float average;
            if (type == ov::element::Type_t::f32) {
                average = findAverage<float>(output);
            } else if (type == ov::element::Type_t::f16) {
                average = findAverage<ov::float16>(output);
            } else {
                ov::nvidia_gpu::throw_ov_exception(std::string{"Unsupported type: "} + type.get_type_name());
            }
            threshold = average * threshold_base;
            std::cout << "threshold = " << threshold << '\n';
        }
        return refOutputs;
    }

protected:
    float threshold_base = 0.0f;
};

/**
 * @brief This class is the actual base class that should be used for the derived test class.
 */
template <typename BaseLayerTest>
class AverageFinder : public BaseLayerTest, public AverageFinderBase {
    static_assert(std::is_base_of_v<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>,
                  "BaseLayerTest should be derived from LayerTestsUtils::LayerTestsCommon");
};

/**
 * @brief This class is the base class for MinMaxAvgFinder class.
 * It is used to find and print min, max, average, min absolute, max absolute and average absolute values for the
 * single layer test class with one output.
 */
class MinMaxAvgFinderBase : virtual public LayerTestsUtils::LayerTestsCommon {
    virtual std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs() override {
        using namespace details;
        const auto refOutputs = LayerTestsCommon::CalculateRefs();
        if (refOutputs.size() == 1) {
            const auto& type = refOutputs[0].first;
            const auto& output = refOutputs[0].second;
            BlobLimits bl;
            if (type == ov::element::Type_t::f32) {
                findLimits<float>(output, bl);
            } else if (type == ov::element::Type_t::f16) {
                findLimits<ov::float16>(output, bl);
            } else {
                ov::nvidia_gpu::throw_ov_exception(std::string{"Unsupported type: "} + type.get_type_name());
            }
        }
        return refOutputs;
    }
};

/**
 * @brief This class is the actual base class that should be used for the derived test class.
 */
template <typename BaseLayerTest>
class MinMaxAvgFinder : public BaseLayerTest, public MinMaxAvgFinderBase {
    static_assert(std::is_base_of_v<LayerTestsUtils::LayerTestsCommon, BaseLayerTest>,
                  "BaseLayerTest should be derived from LayerTestsUtils::LayerTestsCommon");
};

}  // namespace LayerTestsDefinitions
