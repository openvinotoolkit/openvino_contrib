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

typedef std::tuple<EltwiseTestParams, OperationMode> CudaEltwiseTestParams;

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

private:
    std::string secondary_input_name;
};

}  // namespace LayerTestsDefinitions
