// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

namespace ov {
namespace test {
namespace nvidia_gpu {

class ConvertColorCUDALayerTest : virtual public ov::test::SubgraphBaseTest {
private:
    ov::Tensor generate_input(const ov::element::Type element_type, const ov::Shape& shape) const {
        return utils::create_and_fill_tensor(element_type, shape, 255);
    };
    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();
        for (int i = 0; i < func_inputs.size(); ++i) {
            const auto& param = func_inputs[i];
            auto tensor = generate_input(param.get_element_type(), target_input_static_shapes[i]);
            inputs.insert({param.get_node_shared_ptr(), tensor});
        }
    }
};
} // nvidia_gpu
} // test
} // ov
