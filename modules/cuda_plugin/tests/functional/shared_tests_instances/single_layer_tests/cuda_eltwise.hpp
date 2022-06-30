// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <shared_test_classes/single_layer/eltwise.hpp>
#include <string>
#include <tuple>

#include "finite_comparer.hpp"

namespace LayerTestsDefinitions {

enum class OperationMode { NORMAL, PYTHON_DIVIDE };

typedef std::tuple<ov::test::subgraph::EltwiseTestParams, OperationMode> CudaEltwiseTestParams;

/**
 * @brief This class inlcudes some features missing in EltwiseLayerTest class.
 * It allows to run testing on negative values, which is crucial for some elementwise operations.
 * It has the logics of omiting of '0' values for the second input of Divide and Mod operations as reference
 * implementations doesn't support them.
 * Both general and 'Python' modes are also supported for the Divide operation.
 * Unfortunately, some code had to be copied from EltwiseLayerTest to implement logics not supported by polymorphism.
 */

class CudaEltwiseLayerTest : public testing::WithParamInterface<CudaEltwiseTestParams>,
                             virtual public FiniteLayerComparer {
public:
    static constexpr int start_from = -10;
    static constexpr int up_to = 10;
    static constexpr int seed = 1;
    static constexpr int range = up_to - start_from;
    static constexpr int resolution = 1;

    static std::string getTestCaseName(testing::TestParamInfo<CudaEltwiseTestParams> obj);

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;

    void SetUp() override;

    void init_input_shapes(const std::vector<ov::test::InputShape>& shapes) {
        if (shapes.empty()) {
            targetStaticShapes = {{}};
            return;
        }
        size_t targetStaticShapeSize = shapes.front().second.size();
        for (size_t i = 1; i < shapes.size(); ++i) {
            if (targetStaticShapeSize < shapes[i].second.size()) {
                targetStaticShapeSize = shapes[i].second.size();
            }
        }
        targetStaticShapes.resize(targetStaticShapeSize);

        for (const auto& shape : shapes) {
            auto dynShape = shape.first;
            if (dynShape.rank() == 0) {
                ASSERT_EQ(targetStaticShapeSize, 1) << "Incorrect number of static shapes for static case";
                dynShape = shape.second.front();
            }
            inputDynamicShapes.push_back(dynShape);
            for (size_t i = 0; i < targetStaticShapeSize; ++i) {
                targetStaticShapes[i].push_back(i < shape.second.size() ? shape.second.at(i) : shape.second.back());
            }
        }
    }

    void transformInputShapesAccordingEltwise(const ov::PartialShape& secondInputShape) {
        // propagate shapes in case 1 shape is defined
        if (inputDynamicShapes.size() == 1) {
            inputDynamicShapes.push_back(inputDynamicShapes.front());
            for (auto& staticShape : targetStaticShapes) {
                staticShape.push_back(staticShape.front());
            }
        }
        ASSERT_EQ(inputDynamicShapes.size(), 2) << "Incorrect inputs number!";
        if (!secondInputShape.is_static()) {
            return;
        }
        if (secondInputShape.get_shape() == ov::Shape{1}) {
            inputDynamicShapes[1] = secondInputShape;
            for (auto& staticShape : targetStaticShapes) {
                staticShape[1] = secondInputShape.get_shape();
            }
        }
    }

private:
    ov::AnyMap configuration;

    std::vector<ov::PartialShape> inputDynamicShapes;
    std::vector<std::vector<ov::Shape>> targetStaticShapes;

    std::string secondary_input_name;
};

}  // namespace LayerTestsDefinitions
