// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

namespace CUDAPlugin {
namespace kernel {

template <typename Int, Int First, Int... Indices, typename TypeT, typename Switch, typename... Args>
constexpr decltype(auto)
templateSwitch(std::integer_sequence<Int, First, Indices...>, TypeT v, Switch&& switch_, Args&&... args) noexcept(
    noexcept(std::forward<Switch>(switch_).template case_<static_cast<TypeT>(First)>(std::forward<Args>(args)...),
             templateSwitch(std::integer_sequence<Int, Indices...>{},
                            v,
                            std::forward<Switch>(switch_),
                            std::forward<Args>(args)...))) {
    if (static_cast<Int>(v) == First)
        return std::forward<Switch>(switch_).template case_<static_cast<TypeT>(First)>(std::forward<Args>(args)...);
    return templateSwitch(
        std::integer_sequence<Int, Indices...>{}, v, std::forward<Switch>(switch_), std::forward<Args>(args)...);
}

template <typename Int, typename TypeT, typename Switch, typename... Args>
constexpr decltype(auto) templateSwitch(std::integer_sequence<Int>, TypeT v, Switch&& switch_, Args&&... args) noexcept(
    noexcept(switch_.default_(v, std::forward<Args>(args)...))) {
    return std::forward<Switch>(switch_).default_(v, std::forward<Args>(args)...);
}

// example usage:
// struct SimpleSwitch {
//    template <int I>
//    constexpr int case_(int n) const noexcept {
//        return I * n;
//    }
//    constexpr int default_(int, int n) const noexcept { return -1 * n; }
//};
//
// auto n = templateSwitch(std::integer_sequence<int, 1, 2, 4, 6>{}, 4, SimpleSwitch{}, 2);

}  // namespace kernel
}  // namespace CUDAPlugin
