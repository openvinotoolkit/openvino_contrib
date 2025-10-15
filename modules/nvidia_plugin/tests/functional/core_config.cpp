// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_test_constants.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

void core_configuration(ov::test::SubgraphBaseTest* test) {
    ov::element::Type hint = ov::element::f32;
    for (auto& param : test->function->get_parameters()) {
        if (param->get_output_element_type(0) == ov::element::f16) {
            hint = ov::element::f16;
            break;
        }
    }
    // Set inference_precision hint to run fp32 model in fp32 runtime precision as default plugin execution precision
    // may vary
    test->core->set_property(ov::test::utils::DEVICE_NVIDIA,
                             {{ov::hint::inference_precision.name(), hint.get_type_name()}});
}

}  // namespace test
}  // namespace ov
