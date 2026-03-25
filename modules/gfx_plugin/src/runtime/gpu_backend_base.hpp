// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <algorithm>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "openvino/core/except.hpp"
#include "runtime/gpu_types.hpp"

namespace ov {
namespace gfx_plugin {

struct KernelArg {
    enum class Kind { Buffer, Bytes };
    Kind kind = Kind::Buffer;
    uint32_t index = 0;
    size_t offset = 0;
    GpuBuffer buffer{};
    const void* bytes = nullptr;
    size_t byte_size = 0;
};

struct KernelSignature {
    uint32_t arg_count = 0;
};

inline KernelArg make_buffer_arg(uint32_t index, const GpuBuffer& buffer, size_t offset = 0) {
    KernelArg arg;
    arg.kind = KernelArg::Kind::Buffer;
    arg.index = index;
    arg.offset = offset;
    arg.buffer = buffer;
    return arg;
}

inline KernelArg make_bytes_arg(uint32_t index, const void* data, size_t size) {
    KernelArg arg;
    arg.kind = KernelArg::Kind::Bytes;
    arg.index = index;
    arg.bytes = data;
    arg.byte_size = size;
    return arg;
}

struct KernelDispatch {
    size_t grid[3] = {1, 1, 1};
    size_t threads_per_group[3] = {1, 1, 1};
};

struct KernelExecutionHooks {
    std::function<void(GpuCommandEncoderHandle)> on_begin;
    std::function<void(GpuCommandEncoderHandle)> on_end;
    std::function<void()> on_complete;
};

inline uint32_t kernel_args_count(const std::vector<KernelArg>& args) {
    uint32_t max_index = 0;
    bool found = false;
    for (const auto& arg : args) {
        if (arg.kind != KernelArg::Kind::Buffer && arg.kind != KernelArg::Kind::Bytes) {
            continue;
        }
        max_index = std::max(max_index, arg.index);
        found = true;
    }
    return found ? (max_index + 1u) : 0u;
}

inline bool kernel_args_dense(const std::vector<KernelArg>& args, uint32_t* out_count = nullptr) {
    const uint32_t count = kernel_args_count(args);
    if (out_count) {
        *out_count = count;
    }
    if (count == 0) {
        return true;
    }
    std::vector<bool> seen(count, false);
    for (const auto& arg : args) {
        if (arg.kind != KernelArg::Kind::Buffer && arg.kind != KernelArg::Kind::Bytes) {
            continue;
        }
        if (arg.index >= count) {
            return false;
        }
        if (seen[arg.index]) {
            return false;
        }
        seen[arg.index] = true;
    }
    return std::all_of(seen.begin(), seen.end(), [](bool v) { return v; });
}

inline uint32_t ensure_kernel_args_dense(const std::vector<KernelArg>& args, const char* label) {
    uint32_t count = 0;
    OPENVINO_ASSERT(kernel_args_dense(args, &count),
                    label ? label : "GFX",
                    ": kernel args must be densely indexed from 0");
    return count;
}

class KernelBindingPlan {
public:
    explicit KernelBindingPlan(uint32_t arg_count = 0) : m_arg_count(arg_count) {}

    uint32_t arg_count() const {
        return m_arg_count;
    }

    uint32_t resolve_runtime_arg_count(const std::vector<KernelArg>& args, const char* label) const {
        const uint32_t runtime_count = ensure_kernel_args_dense(args, label);
        if (m_arg_count == 0) {
            return runtime_count;
        }
        OPENVINO_ASSERT(runtime_count == m_arg_count,
                        label ? label : "GFX",
                        ": kernel arg count mismatch (expected ",
                        m_arg_count,
                        ", got ",
                        runtime_count,
                        ")");
        return m_arg_count;
    }

private:
    uint32_t m_arg_count = 0;
};

struct KernelBufferBinding {
    GpuBuffer buffer{};
    size_t offset = 0;

    bool operator==(const KernelBufferBinding& other) const {
        const uint64_t lhs_id = buffer.allocation_uid != 0
                                    ? buffer.allocation_uid
                                    : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer.buffer));
        const uint64_t rhs_id = other.buffer.allocation_uid != 0
                                    ? other.buffer.allocation_uid
                                    : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(other.buffer.buffer));
        return lhs_id == rhs_id && buffer.size == other.buffer.size && offset == other.offset;
    }
};

struct KernelBindingTable {
    std::vector<KernelBufferBinding> buffers;

    bool operator==(const KernelBindingTable& other) const {
        return buffers == other.buffers;
    }
};

struct KernelBindingTableHash {
    size_t operator()(const KernelBindingTable& table) const {
        size_t seed = table.buffers.size();
        for (const auto& binding : table.buffers) {
            const uint64_t identity = binding.buffer.allocation_uid != 0
                                          ? binding.buffer.allocation_uid
                                          : static_cast<uint64_t>(reinterpret_cast<uintptr_t>(binding.buffer.buffer));
            const auto handle_hash = std::hash<uint64_t>{}(identity);
            const auto size_hash = std::hash<size_t>{}(binding.buffer.size);
            const auto offset_hash = std::hash<size_t>{}(binding.offset);
            seed ^= handle_hash + 0x9e3779b9u + (seed << 6) + (seed >> 2);
            seed ^= size_hash + 0x9e3779b9u + (seed << 6) + (seed >> 2);
            seed ^= offset_hash + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct PreparedBindingCacheKey {
    GpuBackend backend = GpuBackend::Metal;
    uintptr_t device = 0;
    uint32_t arg_count = 0;

    bool operator==(const PreparedBindingCacheKey& other) const {
        return backend == other.backend && device == other.device && arg_count == other.arg_count;
    }
};

struct PreparedBindingCacheKeyHash {
    size_t operator()(const PreparedBindingCacheKey& key) const {
        auto hash_combine = [](uint64_t seed, uint64_t value) {
            return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
        };
        uint64_t seed = 0;
        seed = hash_combine(seed, static_cast<uint64_t>(key.backend));
        seed = hash_combine(seed, static_cast<uint64_t>(key.device));
        seed = hash_combine(seed, key.arg_count);
        return static_cast<size_t>(seed);
    }
};

class PreparedKernelBindings {
public:
    explicit PreparedKernelBindings(KernelBindingTable table) : m_table(std::move(table)) {}
    virtual ~PreparedKernelBindings() = default;

    const KernelBindingTable& binding_table() const {
        return m_table;
    }

    template <typename T, typename Factory>
    std::shared_ptr<T> get_or_create_backend_state(uintptr_t state_key, Factory&& factory) const {
        {
            std::lock_guard<std::mutex> lock(m_backend_state_mutex);
            if (auto it = m_backend_states.find(state_key); it != m_backend_states.end()) {
                return std::static_pointer_cast<T>(it->second);
            }
        }

        auto created = std::forward<Factory>(factory)();
        OPENVINO_ASSERT(created, "GFX: backend state factory returned null");

        std::lock_guard<std::mutex> lock(m_backend_state_mutex);
        auto& slot = m_backend_states[state_key];
        if (slot) {
            return std::static_pointer_cast<T>(slot);
        }
        slot = created;
        return created;
    }

private:
    KernelBindingTable m_table;
    mutable std::mutex m_backend_state_mutex;
    mutable std::unordered_map<uintptr_t, std::shared_ptr<void>> m_backend_states;
};

class SharedPreparedBindingCache {
public:
    static constexpr size_t kMaxEntries = 512;

    struct Entry {
        std::shared_ptr<const PreparedKernelBindings> bindings;
        std::list<KernelBindingTable>::iterator lru_it;
    };

    std::mutex mutex;
    std::unordered_map<KernelBindingTable, Entry, KernelBindingTableHash> entries;
    std::list<KernelBindingTable> lru;
};

class PreparedBindingCacheRegistry {
public:
    static PreparedBindingCacheRegistry& instance() {
        static PreparedBindingCacheRegistry registry;
        return registry;
    }

    std::shared_ptr<void> acquire(GpuBackend backend, uintptr_t device, uint32_t arg_count) {
        if (arg_count == 0) {
            return std::make_shared<SharedPreparedBindingCache>();
        }
        const PreparedBindingCacheKey key{backend, device, arg_count};
        std::lock_guard<std::mutex> lock(m_mutex);
        if (auto it = m_caches.find(key); it != m_caches.end()) {
            if (auto cache = it->second.lock()) {
                return cache;
            }
        }
        auto cache = std::make_shared<SharedPreparedBindingCache>();
        m_caches[key] = cache;
        return cache;
    }

private:
    std::mutex m_mutex;
    std::unordered_map<PreparedBindingCacheKey,
                       std::weak_ptr<void>,
                       PreparedBindingCacheKeyHash>
        m_caches;
};

inline std::shared_ptr<void> acquire_shared_prepared_binding_cache(GpuBackend backend,
                                                                   uintptr_t device,
                                                                   uint32_t arg_count) {
    return PreparedBindingCacheRegistry::instance().acquire(backend, device, arg_count);
}

inline KernelBindingTable materialize_runtime_bindings(const KernelBindingPlan& plan,
                                                       const std::vector<KernelArg>& args,
                                                       const char* label) {
    const uint32_t binding_count = plan.resolve_runtime_arg_count(args, label);
    KernelBindingTable table;
    table.buffers.resize(binding_count);
    for (const auto& arg : args) {
        OPENVINO_ASSERT(arg.kind == KernelArg::Kind::Buffer,
                        label ? label : "GFX",
                        ": bytes arguments must be materialized into buffers");
        OPENVINO_ASSERT(arg.index < table.buffers.size(),
                        label ? label : "GFX",
                        ": kernel arg index ",
                        arg.index,
                        " exceeds binding ABI size ",
                        table.buffers.size());
        table.buffers[arg.index] = KernelBufferBinding{arg.buffer, arg.offset};
    }
    return table;
}

inline KernelBindingTable materialize_kernel_binding_table(const std::vector<KernelArg>& args, const char* label) {
    const uint32_t arg_count = ensure_kernel_args_dense(args, label);
    KernelBindingTable table;
    table.buffers.resize(arg_count);
    for (const auto& arg : args) {
        OPENVINO_ASSERT(arg.kind == KernelArg::Kind::Buffer,
                        label ? label : "GFX",
                        ": bytes arguments must be materialized into buffers");
        table.buffers[arg.index] = KernelBufferBinding{arg.buffer, arg.offset};
    }
    return table;
}

class ICompiledKernel {
public:
    virtual ~ICompiledKernel() = default;
    virtual uint32_t args_count() const { return 0; }
    virtual void set_args_count(uint32_t /*count*/) {}
    virtual size_t clamp_threadgroup_size(size_t desired) const = 0;
    // Return a kernel instance that can safely participate in another infer/model.
    // Immutable kernels may return themselves; kernels with mutable execution state
    // must return a fresh wrapper sharing only immutable program artifacts.
    virtual std::shared_ptr<ICompiledKernel> fork() const = 0;
    virtual void on_submission_complete() {}
    virtual void execute(GpuCommandBufferHandle command_buffer,
                         const KernelDispatch& dispatch,
                         const std::vector<KernelArg>& args,
                         const KernelExecutionHooks* hooks = nullptr) = 0;
};

class CompiledKernelBase : public ICompiledKernel {
public:
    explicit CompiledKernelBase(uint32_t arg_count = 0)
        : m_binding_plan(std::make_shared<KernelBindingPlan>(arg_count)),
          m_prepared_binding_cache(std::make_shared<SharedPreparedBindingCache>()) {}

    explicit CompiledKernelBase(std::shared_ptr<const KernelBindingPlan> binding_plan)
        : m_binding_plan(binding_plan ? std::move(binding_plan)
                                     : std::make_shared<KernelBindingPlan>()),
          m_prepared_binding_cache(std::make_shared<SharedPreparedBindingCache>()) {}

    CompiledKernelBase(std::shared_ptr<const KernelBindingPlan> binding_plan,
                       std::shared_ptr<void> prepared_binding_cache)
        : m_binding_plan(binding_plan ? std::move(binding_plan)
                                     : std::make_shared<KernelBindingPlan>()),
          m_prepared_binding_cache(prepared_binding_cache ? std::move(prepared_binding_cache)
                                                         : std::make_shared<SharedPreparedBindingCache>()) {}

    uint32_t args_count() const override {
        return m_binding_plan->arg_count();
    }

    void set_args_count(uint32_t count) override {
        if (count == 0) {
            return;
        }
        if (m_binding_plan->arg_count() == 0) {
            m_binding_plan = std::make_shared<KernelBindingPlan>(count);
            return;
        }
        OPENVINO_ASSERT(m_binding_plan->arg_count() == count,
                        "GFX: kernel arg count mismatch (expected ",
                        m_binding_plan->arg_count(),
                        ", got ",
                        count,
                        ")");
    }

protected:
    uint32_t resolve_runtime_arg_count(const std::vector<KernelArg>& args, const char* label) const {
        return m_binding_plan->resolve_runtime_arg_count(args, label);
    }

    uint32_t binding_abi_count() const {
        return m_binding_plan->arg_count();
    }

    KernelBindingTable materialize_runtime_bindings(const std::vector<KernelArg>& args, const char* label) const {
        return gfx_plugin::materialize_runtime_bindings(*m_binding_plan, args, label);
    }

    std::shared_ptr<const PreparedKernelBindings> get_or_create_prepared_bindings(const std::vector<KernelArg>& args,
                                                                                  const char* label) const {
        return get_or_create_prepared_bindings(materialize_runtime_bindings(args, label));
    }

    const std::shared_ptr<const KernelBindingPlan>& binding_plan() const {
        return m_binding_plan;
    }

    const std::shared_ptr<void>& prepared_binding_cache() const {
        return m_prepared_binding_cache;
    }

    virtual std::shared_ptr<const PreparedKernelBindings> create_prepared_bindings(
        const KernelBindingTable& bindings) const {
        return std::make_shared<PreparedKernelBindings>(bindings);
    }

private:
    std::shared_ptr<const PreparedKernelBindings> get_or_create_prepared_bindings(
        const KernelBindingTable& bindings) const {
        auto* cache = static_cast<SharedPreparedBindingCache*>(m_prepared_binding_cache.get());
        OPENVINO_ASSERT(cache, "GFX: prepared binding cache is null");
        {
            std::lock_guard<std::mutex> lock(cache->mutex);
            if (auto it = cache->entries.find(bindings); it != cache->entries.end()) {
                cache->lru.splice(cache->lru.begin(), cache->lru, it->second.lru_it);
                return it->second.bindings;
            }
        }

        auto created = create_prepared_bindings(bindings);
        OPENVINO_ASSERT(created, "GFX: create_prepared_bindings returned null");

        std::lock_guard<std::mutex> lock(cache->mutex);
        if (auto it = cache->entries.find(created->binding_table()); it != cache->entries.end()) {
            cache->lru.splice(cache->lru.begin(), cache->lru, it->second.lru_it);
            return it->second.bindings;
        }
        auto lru_it = cache->lru.insert(cache->lru.begin(), created->binding_table());
        cache->entries.emplace(*lru_it, SharedPreparedBindingCache::Entry{created, lru_it});
        while (cache->entries.size() > SharedPreparedBindingCache::kMaxEntries) {
            const auto& evict_key = cache->lru.back();
            cache->entries.erase(evict_key);
            cache->lru.pop_back();
        }
        return created;
    }

    std::shared_ptr<const KernelBindingPlan> m_binding_plan;
    std::shared_ptr<void> m_prepared_binding_cache;
};

}  // namespace gfx_plugin
}  // namespace ov
