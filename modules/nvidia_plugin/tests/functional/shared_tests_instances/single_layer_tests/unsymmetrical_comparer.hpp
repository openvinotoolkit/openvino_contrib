// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace LayerTestsDefinitions {

template <class BaseLayerTest>
class UnsymmetricalComparer : public BaseLayerTest, virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info, int seed) const {
        return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 10, 0, 1, seed);
    };

    void GenerateInputs() override {
        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        const int baseSeed = 1;

        for (int i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

            const auto& info = infoIt->second;
            auto blob = GenerateInput(*info, baseSeed + i);
            inputs.push_back(blob);
        }
    }
};

}  // namespace LayerTestsDefinitions
