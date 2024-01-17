// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <error.hpp>
#include <functional>
#include <utility>

namespace ov {
namespace nvidia_gpu {

class CancellationToken {
public:
    /**
     * Constructor
     * @param callback Callback that will be called on token cancelled check
     */
    explicit CancellationToken(std::function<void()>&& cancelCallback = nullptr)
        : cancel_callback_{std::move(cancelCallback)} {}

    /**
     * Set token status as cancelled
     */
    void cancel() {
        if (cancel_callback_) {
            cancel_callback_();
        };
    }

private:
    std::function<void()> cancel_callback_;
};

}  // namespace nvidia_gpu
}  // namespace ov
