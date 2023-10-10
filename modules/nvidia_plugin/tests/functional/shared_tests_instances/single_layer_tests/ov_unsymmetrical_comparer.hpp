// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace test {

template <class BaseLayerTest>
class UnsymmetricalComparer : public BaseLayerTest, virtual public ov::test::SubgraphBaseTest {
protected:
    ov::Tensor generate_input(const ov::element::Type element_type, const ov::Shape& shape, int seed) const {
        return ov::test::utils::create_and_fill_tensor(element_type, shape, 10, 0, 1, seed);
    };

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();
        const int base_seed = 1;

        for (int i = 0; i < func_inputs.size(); ++i) {

            const auto& param = func_inputs[i];
            auto tensor = generate_input(param.get_element_type(), target_input_static_shapes[i], base_seed + i);
            inputs.insert({param.get_node_shared_ptr(), tensor});
        }
    }
};
}  // namespace test
}  // namespace ov
