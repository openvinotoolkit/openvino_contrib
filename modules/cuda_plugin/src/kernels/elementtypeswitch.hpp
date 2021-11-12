// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "switch.hpp"

namespace CUDAPlugin {
namespace kernel {

template <Type_t... Types>
struct ElementTypesSwitch {
    static constexpr std::integer_sequence<Type_t, Types...> indices{};
    template <Type_t I, typename Switch, typename... Args>
    constexpr decltype(auto) case_(Switch&& switch_, Args&&... args) const noexcept(
        noexcept(std::forward<Switch>(switch_).template case_<cuda_type_traits_t<I>>(std::forward<Args>(args)...))) {
        return std::forward<Switch>(switch_).template case_<cuda_type_traits_t<I>>(std::forward<Args>(args)...);
    }
    template <typename Switch, typename... Args>
    constexpr decltype(auto) default_(Type_t t, Switch&& switch_, Args&&... args) const
        noexcept(noexcept(std::forward<Switch>(switch_).default_(t, std::forward<Args>(args)...))) {
        return std::forward<Switch>(switch_).default_(t, std::forward<Args>(args)...);
    }
    template <typename Switch, typename... Args>
    static constexpr decltype(auto) switch_(Type_t v, Switch&& switchObj, Args&&... args) noexcept(
        noexcept(templateSwitch(
            indices, v, ElementTypesSwitch{}, std::forward<Switch>(switchObj), std::forward<Args>(args)...))) {
        return templateSwitch(
            indices, v, ElementTypesSwitch{}, std::forward<Switch>(switchObj), std::forward<Args>(args)...);
    }
};

using AllElementTypesSwitch = ElementTypesSwitch<Type_t::boolean,
#if CUDA_VERSION >= 11000
                                                 Type_t::bf16,
#endif
                                                 Type_t::f16,
                                                 Type_t::f32,
                                                 Type_t::f64,
                                                 Type_t::i8,
                                                 Type_t::i16,
                                                 Type_t::i32,
                                                 Type_t::i64,
                                                 Type_t::u8,
                                                 Type_t::u16,
                                                 Type_t::u32,
                                                 Type_t::u64>;

}  // namespace kernel
}  // namespace CUDAPlugin
