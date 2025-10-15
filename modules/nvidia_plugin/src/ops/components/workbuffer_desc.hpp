// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "memory_manager/cuda_workbuffers.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Helps operation to handle multiple workbuffers. Allows to easily skip
 * some allocations without the need to track workbuffer indices.
 */
class WorkbufferDesc {
public:
    WorkbufferDesc() : bsize_{0}, index_{-1} {}
    void addRequest(std::vector<WorkbufferRequest::size_in_bytes_t>& workbuffers_sizes, size_t requested_bsize) {
        if (requested_bsize > 0) {
            bsize_ = requested_bsize;
            index_ = workbuffers_sizes.size();
            workbuffers_sizes.emplace_back(bsize_);
        }
    }
    size_t size() const { return bsize_; }
    template <typename T>
    T* optionalPtr(const std::vector<CUDA::DevicePointer<T*>>& buffers) const {
        if (index_ < 0) {
            return nullptr;
        }
        return requiredPtr(buffers);
    }
    template <typename T>
    T* requiredPtr(const std::vector<CUDA::DevicePointer<T*>>& buffers) const {
        OPENVINO_ASSERT((index_ >= 0) && (index_ < buffers.size()));
        return buffers[index_].get();
    }
    operator bool() const { return index_ >= 0; }

private:
    size_t bsize_;
    int index_;
};

}  // namespace nvidia_gpu
}  // namespace ov
