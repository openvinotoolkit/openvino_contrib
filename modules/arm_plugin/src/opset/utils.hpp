// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <ngraph/node.hpp>
#include <ngraph/variant.hpp>
#include "arm_compute/core/QuantizationInfo.h"
#include "arm_compute/core/Types.h"

#ifndef OV_NEW_API
#define OV_NGRAPH_NAMESPACE ngraph
#else
#define OV_NGRAPH_NAMESPACE ov
#endif

namespace ArmPlugin {
namespace opset {
float round(const float v);

arm_compute::ActivationLayerInfo makeActivationLayerInfo(ngraph::Node* node);

}  // namespace opset

struct SafeCast {
    const char* file;
    const int line;
    template<typename T, typename Base>
    const T* call(const Base* base) {
        auto* dynamic_casted_ptr = dynamic_cast<const T*>(base);
        OPENVINO_ASSERT(dynamic_casted_ptr != nullptr,
            "In file: ", file, ":", line, "\n",
            "Could not cast base pointer: ", base, "to type ", T::get_type_info_static());
        return dynamic_casted_ptr;
    }
    template<typename T, typename Base>
    const std::shared_ptr<T> call(const std::shared_ptr<Base>& base) {
        auto dynamic_casted_ptr = std::dynamic_pointer_cast<T>(base);
        OPENVINO_ASSERT(dynamic_casted_ptr != nullptr,
            "In file: ", file, ":", line, "\n",
            "Could not cast base pointer: ", base, "to type ", T::get_type_info_static());
        return dynamic_casted_ptr;
    }
    template<typename T>
    const std::shared_ptr<T> call(const ov::Any& any) {
        OPENVINO_ASSERT(any.is<std::shared_ptr<T>>(),
            "In file: ", file, ":", line, "\n",
            "Could not cast any to type ", T::get_type_info_static());
        return any.as<std::shared_ptr<T>>();
    }
};

#define safe_cast SafeCast{__FILE__, __LINE__}.call

}  // namespace ArmPlugin
