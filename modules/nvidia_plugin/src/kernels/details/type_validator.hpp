// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>

#include "element_types_switch.hpp"
#include "error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

inline static void throwTypeNotSupported(Type_t element_type) {
    throwIEException(fmt::format("Element type = {} is not supported.", element_type));
}

template <typename ElementTypes>
class TypeValidator {
public:
    inline static void check(Type_t element_type) { TypeValidator<ElementTypes> tv{element_type}; }

    template <typename T, typename... Args>
    constexpr void case_() const noexcept {}

    template <typename T, typename... Args>
    void default_(T t) const {
        throwTypeNotSupported(t);
    }

private:
    explicit TypeValidator(Type_t element_type) { ElementTypes::switch_(element_type, *this); }
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
