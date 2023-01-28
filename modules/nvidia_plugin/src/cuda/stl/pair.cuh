// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace CUDA {

template <typename K, typename T>
struct Pair {
    K first;
    T second;
};

template <typename K, typename T, typename... TArgs>
__host__ __device__ Pair<K, T> make_pair(K k, T t) {
    return Pair<K, T>{k, t};
}

}  // namespace CUDA
