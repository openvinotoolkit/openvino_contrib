// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <unordered_map>

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Generic LRU (Least Recently Used) cache.
 *
 * O(1) lookup and insertion via hash map + doubly-linked list.
 * Most recently accessed items are at the front of the list.
 * When capacity is exceeded, the least recently used item is evicted.
 *
 * @tparam Key   Cache key type
 * @tparam Value Cached value type
 * @tparam Hash  Hash function for Key (defaults to std::hash<Key>)
 */
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class LruCache {
public:
    explicit LruCache(size_t capacity) : capacity_{capacity} {}

    /**
     * Find a cached value by key.
     * If found, promotes the entry to most-recently-used.
     * @return Pointer to cached value, or nullptr on miss.
     */
    Value* find(const Key& key) {
        auto it = map_.find(key);
        if (it == map_.end()) {
            return nullptr;
        }
        // Move to front (most recently used)
        items_.splice(items_.begin(), items_, it->second);
        return &it->second->second;
    }

    /**
     * Insert or update a cache entry.
     * If at capacity, evicts the least recently used entry.
     * @return Reference to the stored value.
     */
    Value& insert(const Key& key, Value value) {
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Update existing
            it->second->second = std::move(value);
            items_.splice(items_.begin(), items_, it->second);
            return it->second->second;
        }

        // Evict LRU if at capacity
        if (map_.size() >= capacity_) {
            auto& lru = items_.back();
            map_.erase(lru.first);
            items_.pop_back();
        }

        // Insert new at front
        items_.emplace_front(key, std::move(value));
        map_[key] = items_.begin();
        return items_.front().second;
    }

    size_t size() const { return map_.size(); }

    void clear() {
        map_.clear();
        items_.clear();
    }

private:
    size_t capacity_;
    std::list<std::pair<Key, Value>> items_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator, Hash> map_;
};

}  // namespace nvidia_gpu
}  // namespace ov
