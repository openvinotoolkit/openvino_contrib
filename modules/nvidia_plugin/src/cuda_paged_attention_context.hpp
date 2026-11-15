// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "cuda/runtime.hpp"
#include "memory_manager/tensor_types.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Per-request device mirrors of the paged KV caches (key_cache.N / value_cache.N).
 *
 * GenAI owns the caches as host ov::Tensors and modifies them outside the model in two
 * ways: it copies the old contents into a bigger tensor when the cache grows, and it
 * copies blocks for copy-on-write (beam search, parallel sampling, prefix caching).
 * The host tensor is therefore the source of truth, and the device copy is kept coherent
 * with it by the following invariant:
 *
 *   After every infer, every slot written by PagedAttention is identical on host and device.
 *
 * It is maintained by
 *  - bind():            growth keeps the device contents (device-to-device copy); a new or
 *                       desynchronized cache is seeded with a full host-to-device upload;
 *  - refresh_blocks():  before a step appends into a partially filled block, that block is
 *                       re-uploaded from the host (it may be a copy-on-write destination);
 *  - write_back():      every block written by a step is downloaded back to the host. The
 *                       copies are deferred and issued by flush() once all kernels have been
 *                       enqueued, before infer() returns (GenAI reads the host tensor on its
 *                       next schedule()).
 *
 * Owned by CudaInferRequest (so it survives across infer() calls); passed by reference to
 * InferenceRequestContext. All methods are called on the thread that runs the request.
 */
class PagedAttentionContext {
public:
    /**
     * @brief Binds the device mirror of cache `id` to the host tensor bound for this infer
     *        and returns its device pointer.
     */
    void* bind(BufferID id, ov::Tensor& host, const CUDA::Stream& stream) {
        const auto& shape = host.get_shape();
        OPENVINO_ASSERT(!shape.empty(), "PagedAttention: KV cache tensor must have a rank");
        const std::size_t bytes = host.get_byte_size();
        const std::size_t num_blocks = shape[0];

        auto it = caches_.find(id);
        if (it == caches_.end()) {
            it = caches_.emplace(id, Entry{}).first;
        }
        Entry& e = it->second;
        if (bytes > e.capacity) {
            // GenAI grows the cache a few blocks at a time: reserve some headroom when the
            // device has room for it, otherwise allocate exactly what is needed.
            std::size_t capacity = std::max(bytes, e.capacity + e.capacity / 2);
            CUDA::DefaultAllocation alloc{nullptr};
            try {
                alloc = CUDA::DefaultStream::stream().malloc(capacity);
            } catch (const std::exception&) {
                capacity = bytes;
                alloc = CUDA::DefaultStream::stream().malloc(capacity);
            }
            if (e.synced && e.bytes > 0) {
                // GenAI copied the old host contents into the leading part of the new tensor;
                // by the invariant they equal the old device contents.
                stream.transfer(CUDA::DevicePointer<void*>{alloc.get()},
                                CUDA::DevicePointer<const void*>{e.alloc.get()},
                                e.bytes);
                // Free the old buffer right away (only after the copy has read it): GenAI grows
                // all layers at once, so keeping the old buffers until the end of the infer
                // would multiply the peak device memory.
                stream.synchronize();
            }
            e.alloc = std::move(alloc);
            e.capacity = capacity;
        }
        e.bytes = bytes;
        e.block_bytes = num_blocks ? bytes / num_blocks : 0;
        e.host = static_cast<uint8_t*>(host.data());
        if (!e.synced) {
            if (bytes > 0) {
                stream.upload(CUDA::DevicePointer<void*>{e.alloc.get()}, e.host, bytes);
            }
            e.synced = true;
        }
        return e.alloc.get();
    }

    /// Re-uploads the given blocks of cache `id` from the host tensor bound by bind().
    void refresh_blocks(BufferID id, const std::vector<int32_t>& blocks, const CUDA::Stream& stream) {
        const Entry& e = entry(id);
        for_each_run(blocks, [&](int32_t first, int32_t count) {
            const std::size_t offset = static_cast<std::size_t>(first) * e.block_bytes;
            const std::size_t size = static_cast<std::size_t>(count) * e.block_bytes;
            OPENVINO_ASSERT(offset + size <= e.bytes, "PagedAttention: refreshed block is out of the KV cache");
            stream.upload(CUDA::DevicePointer<void*>{static_cast<uint8_t*>(e.alloc.get()) + offset},
                          e.host + offset,
                          size);
        });
    }

    /// Schedules the download of the given blocks of cache `id` to its host tensor (see flush()).
    void write_back(BufferID id, const std::vector<int32_t>& blocks) {
        const Entry& e = entry(id);
        for_each_run(blocks, [&](int32_t first, int32_t count) {
            const std::size_t offset = static_cast<std::size_t>(first) * e.block_bytes;
            const std::size_t size = static_cast<std::size_t>(count) * e.block_bytes;
            OPENVINO_ASSERT(offset + size <= e.bytes, "PagedAttention: written block is out of the KV cache");
            pending_.push_back({e.host + offset, static_cast<const uint8_t*>(e.alloc.get()) + offset, size});
        });
    }

    /// Issues the scheduled downloads. Must be called after all kernels of the infer are enqueued.
    void flush(const CUDA::Stream& stream) {
        for (const auto& p : pending_) {
            stream.download(p.host, CUDA::DevicePointer<const void*>{p.device}, p.bytes);
        }
        pending_.clear();
    }

    /**
     * @brief Marks every device mirror as not matching its host tensor (an infer failed or was
     *        cancelled half-way). The next bind() re-seeds it with a full upload.
     */
    void mark_desynced() {
        for (auto& kv : caches_) {
            kv.second.synced = false;
        }
        pending_.clear();
    }

    bool empty() const { return caches_.empty(); }

private:
    struct Entry {
        CUDA::DefaultAllocation alloc{nullptr};
        std::size_t capacity = 0;     ///< allocated device bytes
        std::size_t bytes = 0;        ///< bytes of the currently bound host tensor
        std::size_t block_bytes = 0;  ///< bytes of one cache block
        uint8_t* host = nullptr;      ///< data of the currently bound host tensor
        bool synced = false;          ///< device holds every slot written on the host
    };
    struct PendingDownload {
        void* host;
        const void* device;
        std::size_t bytes;
    };

    const Entry& entry(BufferID id) const {
        const auto it = caches_.find(id);
        OPENVINO_ASSERT(it != caches_.end(), "PagedAttention: KV cache ", id, " is not bound");
        return it->second;
    }

    /// Calls fn(first, count) for every run of consecutive block ids in a sorted list.
    template <typename Fn>
    static void for_each_run(const std::vector<int32_t>& sorted_blocks, Fn&& fn) {
        for (std::size_t i = 0; i < sorted_blocks.size();) {
            std::size_t j = i + 1;
            while (j < sorted_blocks.size() && sorted_blocks[j] == sorted_blocks[j - 1] + 1) {
                ++j;
            }
            fn(sorted_blocks[i], static_cast<int32_t>(j - i));
            i = j;
        }
    }

    std::unordered_map<BufferID, Entry> caches_;
    std::vector<PendingDownload> pending_;
};

}  // namespace nvidia_gpu
}  // namespace ov
