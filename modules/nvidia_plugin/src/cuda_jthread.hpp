// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>

namespace ov {
namespace nvidia_gpu {

class CudaJThread final {
public:
    template <typename Function, typename... Args>
    explicit CudaJThread(Function&& f, Args&&... args) {
        thread_ = std::thread(std::forward<Function>(f), std::forward<Args>(args)...);
    }
    CudaJThread(CudaJThread&&) noexcept = default;
    CudaJThread& operator=(CudaJThread&&) noexcept = default;
    CudaJThread(const CudaJThread&) noexcept = delete;
    CudaJThread& operator=(const CudaJThread&) noexcept = delete;

    ~CudaJThread() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
};

}  // namespace nvidia_gpu
}  // namespace ov
