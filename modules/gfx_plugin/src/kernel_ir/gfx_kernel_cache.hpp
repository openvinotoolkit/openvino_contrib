// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <chrono>

#include "runtime/gpu_backend_base.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gpu_buffer.hpp"

namespace ov {
namespace gfx_plugin {

inline uint64_t gfx_hash_bytes(const void* data, size_t size) {
    constexpr uint64_t kOffset = 1469598103934665603ull;
    constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t hash = kOffset;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(bytes[i]);
        hash *= kPrime;
    }
    return hash;
}

inline uint64_t gfx_hash_string(const std::string& s) {
    return s.empty() ? 0ull : gfx_hash_bytes(s.data(), s.size());
}

inline uint64_t gfx_hash_combine(uint64_t seed, uint64_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

struct KernelCacheKey {
    GpuBackend backend = GpuBackend::Metal;
    uintptr_t device = 0;
    uint64_t code_hash = 0;
    uint64_t entry_hash = 0;
    uint32_t arg_count = 0;

    bool operator==(const KernelCacheKey& other) const {
        return backend == other.backend &&
               device == other.device &&
               code_hash == other.code_hash &&
               entry_hash == other.entry_hash &&
               arg_count == other.arg_count;
    }
};

struct KernelCacheKeyHash {
    size_t operator()(const KernelCacheKey& key) const {
        uint64_t seed = 0;
        seed = gfx_hash_combine(seed, static_cast<uint64_t>(key.backend));
        seed = gfx_hash_combine(seed, static_cast<uint64_t>(key.device));
        seed = gfx_hash_combine(seed, key.code_hash);
        seed = gfx_hash_combine(seed, key.entry_hash);
        seed = gfx_hash_combine(seed, key.arg_count);
        return static_cast<size_t>(seed);
    }
};

inline KernelCacheKey make_kernel_cache_key(GpuBackend backend,
                                            uintptr_t device,
                                            const void* code,
                                            size_t code_bytes,
                                            const std::string& entry,
                                            uint32_t arg_count) {
    KernelCacheKey key{};
    key.backend = backend;
    key.device = device;
    key.code_hash = (code && code_bytes) ? gfx_hash_bytes(code, code_bytes) : 0ull;
    key.entry_hash = gfx_hash_string(entry);
    key.arg_count = arg_count;
    return key;
}

class GfxKernelCache {
public:
    static GfxKernelCache& instance();

    std::shared_ptr<ICompiledKernel> lookup(const KernelCacheKey& key);
    void store(const KernelCacheKey& key, const std::shared_ptr<ICompiledKernel>& kernel);

private:
    std::mutex m_mutex;
    std::unordered_map<KernelCacheKey,
                       std::weak_ptr<ICompiledKernel>,
                       KernelCacheKeyHash> m_cache;
};

template <typename CompileFn>
inline std::shared_ptr<ICompiledKernel> lookup_or_compile_kernel(GpuBackend backend,
                                                                 uintptr_t device,
                                                                 const void* code,
                                                                 size_t code_bytes,
                                                                 const std::string& entry,
                                                                 uint32_t arg_count,
                                                                 CompileFn&& compile_fn) {
    if (arg_count == 0) {
        return std::forward<CompileFn>(compile_fn)();
    }
    const KernelCacheKey key = make_kernel_cache_key(backend,
                                                     device,
                                                     code,
                                                     code_bytes,
                                                     entry,
                                                     arg_count);
    const auto lookup_start = current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (auto cached = GfxKernelCache::instance().lookup(key)) {
        if (current_compile_trace()) {
            increment_compile_counter("kernel_cache_hit_count");
            add_compile_segment(
                "kernel_cache_lookup_hit",
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - lookup_start)
                                          .count()));
        }
        return cached->fork();
    }
    if (current_compile_trace()) {
        increment_compile_counter("kernel_cache_miss_count");
        add_compile_segment(
            "kernel_cache_lookup_miss",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - lookup_start)
                                      .count()));
    }
    auto kernel = std::forward<CompileFn>(compile_fn)();
    if (kernel) {
        GfxKernelCache::instance().store(key, kernel);
        increment_compile_counter("kernel_cache_store_count");
    }
    return kernel;
}

}  // namespace gfx_plugin
}  // namespace ov
