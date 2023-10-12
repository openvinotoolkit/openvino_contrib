// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <error.hpp>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

namespace details {

template <typename T>
static T find_average(const T* ptr, const size_t size) {
    const auto abs_sum =
        std::accumulate(ptr, ptr + size, float(0.0f), [](float a, T b) {return std::abs(a) + std::abs(static_cast<float>(b)); });
    const T average =  static_cast<T>(abs_sum / size);
    std::cout << "average absolute :" << average << '\n';
    return average;
}

struct BlobLimits {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    float avg = 0.0f;
    float abs_min = std::numeric_limits<float>::max();
    float abs_max = 0.0f;
    float abs_avg = 0.0f;
};

template <typename T>
static BlobLimits find_limits(const T* output, const size_t size, BlobLimits& bl) {
    bl = BlobLimits{};
    const auto* ptr = output;
    float sum = 0.0f;
    float abs_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        const auto el = static_cast<float>(ptr[i]);
        const auto abs_el = std::abs(el);
        bl.min = el < bl.min ? el : bl.min;
        bl.max = el > bl.max ? el : bl.max;
        bl.abs_min = abs_el < bl.abs_min ? abs_el : bl.abs_min;
        bl.abs_max = abs_el > bl.abs_max ? abs_el : bl.abs_max;
        sum += el;
        abs_sum += abs_el;
    }
    bl.avg = sum / size;
    bl.abs_avg = abs_sum / size;

    std::cout << "min = " << bl.min << ", max = " << bl.max << ", avg = " << bl.avg << '\n';
    std::cout << "abs_min = " << bl.abs_min << ", abs_max = " << bl.abs_max << ", abs_avg = " << bl.abs_avg << '\n';

    return bl;
}

}  // namespace details

/**
 * @brief This class is the base class for AverageFinder class.
 * It is used to set the threshold for SubgraphBaseTest::comapare() functions
 * accordingly to the average absolute value of the reference output of a single layer test class.
 * To use it, threshold_base should be set in the derived class.
 * threshold = average * threshold_base
 * For now can be used only for the operations with one output.
 */
class AverageFinderBase : virtual public SubgraphBaseTest {
    virtual std::vector<ov::Tensor> calculate_refs() override {
        using namespace details;
        const auto ref_outputs = SubgraphBaseTest::calculate_refs();
        if (ref_outputs.size() == 1) {
            const auto& type = ref_outputs[0].get_element_type();
            float average;
            if (type == ov::element::Type_t::f32) {
                average = find_average(ref_outputs[0].data<float>(), ref_outputs[0].get_size());
            } else if (type == ov::element::Type_t::f16) {
                average = find_average(ref_outputs[0].data<ov::float16>(), ref_outputs[0].get_size());
            } else {
                ov::nvidia_gpu::throw_ov_exception(std::string{"Unsupported type: "} + type.get_type_name());
            }
            if (!isinf(average))
                abs_threshold = average * threshold_base;
            std::cout << "threshold = " << abs_threshold << '\n';
        }
        return ref_outputs;
    }

protected:
    float threshold_base = 0.0f;
};

/**
 * @brief This class is the actual base class that should be used for the derived test class.
 */
template <typename BaseLayerTest>
class AverageFinder : public BaseLayerTest, public AverageFinderBase {
    static_assert(std::is_base_of_v<SubgraphBaseTest, BaseLayerTest>,
                  "BaseLayerTest should be derived from ov::test::SubgraphBaseTest");
};

/**
 * @brief This class is the base class for MinMaxAvgFinder class.
 * It is used to find and print min, max, average, min absolute, max absolute and average absolute values for the
 * single layer test class with one output.
 */
class MinMaxAvgFinderBase : virtual public SubgraphBaseTest {
    virtual std::vector<ov::Tensor> calculate_refs() override {
        using namespace details;
        const auto ref_outputs = SubgraphBaseTest::calculate_refs();
        if (ref_outputs.size() == 1) {
            const auto& type = ref_outputs[0].get_element_type();
            BlobLimits bl;
            if (type == ov::element::Type_t::f32) {
                find_limits(ref_outputs[0].data<float>(), ref_outputs[0].get_size(), bl);
            } else if (type == ov::element::Type_t::f16) {
                find_limits(ref_outputs[0].data<ov::float16>(), ref_outputs[0].get_size(), bl);
            } else {
                ov::nvidia_gpu::throw_ov_exception(std::string{"Unsupported type: "} + type.get_type_name());
            }
        }
        return ref_outputs;
    }
};

/**
 * @brief This class is the actual base class that should be used for the derived test class.
 */
template <typename BaseLayerTest>
class MinMaxAvgFinder : public BaseLayerTest, public MinMaxAvgFinderBase {
    static_assert(std::is_base_of_v<SubgraphBaseTest, BaseLayerTest>,
                  "BaseLayerTest should be derived from ov::test::SubgraphBaseTest");
};

}  // namespace test
}  // namespace ov
