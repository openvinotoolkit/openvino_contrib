// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <array>

#include "cuda_type_traits.hpp"
#include "error.hpp"
#include "fmt/format.h"

namespace ov {
namespace nvidia_gpu {

namespace kernel {

constexpr size_t type_count = type_t_last_value - type_t_first_value + 1;

enum Dimension : size_t { DIM_1D = 1, DIM_2D, DIM_3D };

// clang-format off
/// \brief By instantiating this template there are created functors TFunctor with all combinations of types.
///        The template structure instance contains a multidimensional matrix of pointers to TFunctor::function of type TFunPtr.
///        These pointers can be accessed by subscript notation.
///
/// For example, if the underlying function need to be templated with 3 template arguments of one of the type
/// cuda_type_traits_t<...>, it must be wrapped by the functor TFunctor like the next.
///
/// template <typename T1, typename T2, typename T3>
/// struct FunctorImpl {
///    static void function(...) {
///        underlying_function<T1, T2, T3>(...);
///    }
/// };
///
/// For that case the three-dimensional matrix of the FunctorImpl<>::function
/// method pointers will be created. Also, the D template argument must be specified as Dimension::DIM_3D.
/// The template structure can be instantiated like the next.
///    static constexpr TypedFunctor<FunctorImpl, TFuncPtr, DIM_3D> instance{};
/// And the function pointer can be accessed like the next.
///    auto func_ptr = instance[type_1][type_2][type_3];
/// Where subscription index (type_1, type_2, type_3) is of type ov::nvidia_gpu::kernel::Type_t enum.
///
/// This template can be used for any dimension number, if the Dimension enumeration extended properly.
/// For trivial case with one dimension the functor will be like the next.
///
/// template <typename T>
/// struct FunctorImpl {
///    static void function(...) {
///        underlying_function<T>(...);
///    }
/// };
/// The template structure can be instantiated like the next.
///    static constexpr TypedFunctor<FunctorImpl, TFuncPtr, DIM_1D> instance{};
/// And the function pointer can be accessed like the next.
///    auto func_ptr = instance[ov::nvidia_gpu::kernel::Type_t::f32];
// clang-format on
template <template <typename... Types> class TFunctor, typename TFunPtr, Dimension D>
struct TypedFunctor : private std::array<TypedFunctor<TFunctor, TFunPtr, Dimension(D - 1)>, type_count> {
    constexpr TypedFunctor() : TypedFunctor(std::make_index_sequence<type_count>(), std::make_index_sequence<0>()) {}

    template <size_t... Indx>
    constexpr TypedFunctor(std::index_sequence<Indx...> index_seq)
        : TypedFunctor(std::make_index_sequence<type_count>(), index_seq) {}

    const auto& operator[](const std::size_t type_idx) const = delete;
    const auto& operator[](const Type_t type) const {
        const size_t type_idx = static_cast<size_t>(type) - type_t_first_value;
        if (type_idx >= this->size()) {
            throwIEException(fmt::format("TypedFunctor[Dimension={}]: Type = {} is not supported by TypedFunctor !!",
                                         DIM_1D,
                                         static_cast<Type_t>(type_idx)));
        }
        return std::array<TypedFunctor<TFunctor, TFunPtr, Dimension(D - 1)>, type_count>::operator[](type_idx);
    }

private:
    template <size_t... I, size_t... Indx>
    constexpr TypedFunctor(std::index_sequence<I...>, std::index_sequence<Indx...>)
        : std::array<TypedFunctor<TFunctor, TFunPtr, Dimension(D - 1)>, type_count>{
              TypedFunctor<TFunctor, TFunPtr, Dimension(D - 1)>(std::index_sequence<Indx..., I>())...} {}
};

template <template <typename... Types> class TFunctor, typename TFunPtr>
struct TypedFunctor<TFunctor, TFunPtr, DIM_1D> : private std::array<TFunPtr, type_count> {
    template <size_t... Indx>
    constexpr TypedFunctor(std::index_sequence<Indx...> index_seq)
        : TypedFunctor(std::make_index_sequence<type_count>(), index_seq) {}

    const auto& operator[](const std::size_t type_idx) const = delete;
    const auto& operator[](const Type_t type) const {
        const size_t type_idx = static_cast<size_t>(type) - type_t_first_value;
        if (type_idx >= this->size()) {
            throwIEException(fmt::format("TypedFunctor[Dimension={}]: Type = {} is not supported by TypedFunctor !!",
                                         DIM_1D,
                                         static_cast<Type_t>(type_idx)));
        }
        return std::array<TFunPtr, type_count>::operator[](type_idx);
    }

private:
    template <size_t... I, size_t... Indx>
    constexpr TypedFunctor(std::index_sequence<I...>, std::index_sequence<Indx...>)
        : std::array<TFunPtr, type_count>{
              &TFunctor<cuda_type_traits_t<static_cast<Type_t>(Indx + type_t_first_value)>...,
                        cuda_type_traits_t<static_cast<Type_t>(I + type_t_first_value)>>::function...} {}
};

}  // namespace kernel

}  // namespace nvidia_gpu
}  // namespace ov
