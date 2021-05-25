// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <functional>
#include <ie_extension.h>
#include <cpp_interfaces/exception2status.hpp>
#include <utility>
#include <ie_common.h>

namespace CUDAPlugin {

class CancellationToken {
 public:
     /**
      * Constructor
      * @param callback Callback that will be called on token cancelled check
      */
    explicit CancellationToken(std::function<void()>&& cancelCallback = nullptr)
        : cancel_callback_{std::move(cancelCallback)} {
    }

    /**
     * Set token status as cancelled
     */
    void Cancel() {
        is_cancelled_.store(true, std::memory_order_release);
    }

    /**
     * Throws exception THROW_IE_EXCEPTION_WITH_STATUS(INFER_CANCELLED) if detected cancel status
     */
    void Check() {
        if (is_cancelled_.load(std::memory_order_acquire)) {
            is_cancelled_.store(false, std::memory_order_release);
            if (cancel_callback_) {
                cancel_callback_();
            }
            THROW_IE_EXCEPTION_WITH_STATUS(INFER_CANCELLED);
        }
    }

 private:
    std::atomic<bool> is_cancelled_{false};
    std::function<void()> cancel_callback_;
};

} // namespace CUDAPlugin
