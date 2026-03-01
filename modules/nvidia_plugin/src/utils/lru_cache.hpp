// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Generic LRU (Least Recently Used) cache backed by a flat vector.
 *
 * For small capacities (typical: 8) linear scan over contiguous memory
 * is faster than std::unordered_map + std::list due to cache locality
 * and zero heap allocations per operation.
 *
 * Most recently accessed items are at the front of the vector.
 * When capacity is exceeded, the least recently used (back) item is evicted.
 *
 * @tparam Key   Cache key type (must support operator==)
 * @tparam Value Cached value type
 * @tparam Hash  Unused, kept for API compatibility
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache {
public:
    explicit LruCache(size_t capacity) : capacity_{capacity} {
        entries_.reserve(capacity);
    }

    /**
     * Find a cached value by key.
     * If found, promotes the entry to most-recently-used (front).
     * @return Pointer to cached value, or nullptr on miss.
     */
    Value* find(const Key& key) {
        auto it = std::find_if(entries_.begin(), entries_.end(),
                               [&key](const auto& e) { return e.first == key; });
        if (it == entries_.end()) {
            return nullptr;
        }
        promote(it);
        return &entries_.front().second;
    }

    /**
     * Insert or update a cache entry.
     * If at capacity, evicts the least recently used entry.
     * @return Reference to the stored value.
     */
    Value& insert(const Key& key, Value value) {
        auto it = std::find_if(entries_.begin(), entries_.end(),
                               [&key](const auto& e) { return e.first == key; });
        if (it != entries_.end()) {
            it->second = std::move(value);
            promote(it);
            return entries_.front().second;
        }

        // Evict LRU if at capacity
        if (entries_.size() >= capacity_) {
            entries_.pop_back();
        }

        // Insert at front
        entries_.emplace(entries_.begin(), key, std::move(value));
        return entries_.front().second;
    }

    size_t size() const { return entries_.size(); }

    void clear() { entries_.clear(); }

private:
    using Iterator = typename std::vector<std::pair<Key, Value>>::iterator;

    void promote(Iterator it) {
        if (it != entries_.begin()) {
            std::rotate(entries_.begin(), it, it + 1);
        }
    }

    const size_t capacity_;
    std::vector<std::pair<Key, Value>> entries_;
};

}  // namespace nvidia_gpu
}  // namespace ov
