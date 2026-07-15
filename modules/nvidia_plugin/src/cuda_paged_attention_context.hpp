// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>

#include "cuda/runtime.hpp"
#include "memory_manager/tensor_types.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Per-request persistent device-resident paged KV caches.
 *
 * PagedAttention's key_cache.N / value_cache.N are model Parameters that GenAI owns
 * as host tensors, but for correctness the K/V written on one step must be visible on
 * the next. Rather than round-trip the (multi-GB) cache to host every step, the plugin
 * keeps a device-resident cache per cache Parameter that persists across inferences of
 * the same request (MVP: single sequence, no host-side block copy/eviction — that is
 * M2).
 *
 * Owned by CudaInferRequest (so it survives across infer() calls); passed by reference
 * to InferenceRequestContext. Mirrors CudaVariableContext.
 */
class PagedAttentionContext {
public:
    /// Returns a device buffer of at least `bytes` for the cache identified by `id`,
    /// allocating (and zero-initializing) it on first use or reallocating on growth.
    /// The zero-init runs on the caller's per-request `stream` (never the default
    /// stream), so it neither serializes concurrent requests device-wide nor races
    /// the paged-attention kernels that run on that same stream.
    void* get_or_allocate(BufferID id, std::size_t bytes, const CUDA::Stream& stream) {
        auto it = caches_.find(id);
        if (it == caches_.end() || it->second.bytes < bytes) {
            Entry e;
            e.bytes = bytes;
            e.alloc = CUDA::DefaultStream::stream().malloc(std::max(bytes, std::size_t{1}));
            if (bytes > 0) {
                stream.memset(CUDA::DevicePointer<void*>{e.alloc.get()}, 0, bytes);
            }
            it = caches_.insert_or_assign(id, std::move(e)).first;
        }
        return it->second.alloc.get();
    }

    bool empty() const { return caches_.empty(); }

private:
    struct Entry {
        CUDA::DefaultAllocation alloc{nullptr};
        std::size_t bytes = 0;
    };
    std::unordered_map<BufferID, Entry> caches_;
};

}  // namespace nvidia_gpu
}  // namespace ov
