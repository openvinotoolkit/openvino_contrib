// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <future>

namespace ov {
namespace nvidia_gpu {

/// Temporary replacement for `std::latch` (until C++20)
class CudaLatch final {
public:
    explicit CudaLatch(std::ptrdiff_t expected) : counter_{expected} {}

    void wait() { p_.get_future().wait(); }

    void count_down() {
        if (--counter_ == 0) p_.set_value();
    }

private:
    std::atomic<std::ptrdiff_t> counter_;
    std::promise<void> p_;
};

}  // namespace nvidia_gpu
}  // namespace ov
