// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/device_pointers.hpp>
#include <gsl/pointers>
#include <memory>
#include <memory_manager/tensor_types.hpp>
#include <unordered_map>
#include <vector>

namespace ov {
namespace nvidia_gpu {

/**
 * @brief MemoryModel describes a size of continous memory block on CUDA device
 * and a location of every tensor within this block.
 */
class MemoryModel {
public:
    using Ptr = std::shared_ptr<MemoryModel>;

    /**
     * @param [in] bsize Memory block size in bytes.
     * @param [in] offsets Maps buffer identifiers to tensor offsets within a memory block.
     */
    MemoryModel(size_t bsize, const std::unordered_map<BufferID, ptrdiff_t>& offsets);

    /**
     * @returns The size of memory block
     */
    size_t deviceMemoryBlockSize() const;

    void* deviceBufferPtr(CUDA::DevicePointer<uint8_t*> devPtr, const BufferID& id);
    void* deviceTensorPtr(CUDA::DevicePointer<uint8_t*> devPtr, const TensorID& id);

    /**
     * Provides buffer memory offset if any.
     *
     * @param [in] id Buffer identifier.
     * @param [out] offset Tensor memory offset with respect to the beginning of the block.
     * @returns false if memory block doesn't contain a tensor with requested identifier.
     */
    bool offsetForBuffer(BufferID id, ptrdiff_t& offset) const;

    const std::vector<BufferID>& bufferIds() const { return buffer_ids_; }

private:
    size_t bsize_;
    std::vector<BufferID> buffer_ids_;
    std::unordered_map<BufferID, ptrdiff_t> offsets_;
};

}  // namespace nvidia_gpu
}  // namespace ov
