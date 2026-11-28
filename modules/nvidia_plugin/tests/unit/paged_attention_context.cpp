// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstring>
#include <numeric>

#include "cuda_paged_attention_context.hpp"

using ov::nvidia_gpu::PagedAttentionContext;

namespace {

// Cache layout [num_blocks, heads=1, block_size=2, head_size=2] (f32): one block = 4 floats.
constexpr size_t kBlockElems = 4;

ov::Tensor make_cache(size_t num_blocks, float first_value) {
    ov::Tensor t(ov::element::f32, ov::Shape{num_blocks, 1, 2, 2});
    std::iota(t.data<float>(), t.data<float>() + t.get_size(), first_value);
    return t;
}

std::vector<float> device_contents(const void* device, size_t elems, const CUDA::Stream& stream) {
    std::vector<float> v(elems);
    stream.download(v.data(), CUDA::DevicePointer<const void*>{device}, elems * sizeof(float));
    stream.synchronize();
    return v;
}

std::vector<float> host_contents(const ov::Tensor& t) {
    return {t.data<const float>(), t.data<const float>() + t.get_size()};
}

float* block(ov::Tensor& t, size_t b) { return t.data<float>() + b * kBlockElems; }

class PagedAttentionContextTest : public testing::Test {
protected:
    CUDA::Stream stream;
    PagedAttentionContext ctx;
    static constexpr ov::nvidia_gpu::BufferID kId = 42;
};

}  // namespace

TEST_F(PagedAttentionContextTest, FirstBindSeedsDeviceFromHost) {
    auto host = make_cache(3, 1.0f);
    const void* dev = ctx.bind(kId, host, stream);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), host_contents(host));
}

TEST_F(PagedAttentionContextTest, WriteBackUpdatesOnlyScheduledBlocksAfterFlush) {
    auto host = make_cache(3, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    // Simulate reshape_and_cache writing blocks 0 and 2 on the device.
    const std::vector<float> written(3 * kBlockElems, -5.0f);
    stream.upload(CUDA::DevicePointer<void*>{dev}, written.data(), written.size() * sizeof(float));
    const auto before = host_contents(host);

    ctx.write_back(kId, {0, 2});
    EXPECT_EQ(host_contents(host), before);  // deferred until flush()
    ctx.flush(stream);
    stream.synchronize();

    auto expected = before;
    std::fill_n(expected.begin(), kBlockElems, -5.0f);
    std::fill_n(expected.begin() + 2 * kBlockElems, kBlockElems, -5.0f);
    EXPECT_EQ(host_contents(host), expected);
}

TEST_F(PagedAttentionContextTest, GrowthKeepsExistingDeviceContents) {
    auto host = make_cache(2, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    // A step writes block 1 on the device and writes it back.
    const std::vector<float> written(kBlockElems, 7.0f);
    stream.upload(CUDA::DevicePointer<void*>{static_cast<float*>(dev) + kBlockElems},
                  written.data(),
                  written.size() * sizeof(float));
    ctx.write_back(kId, {1});
    ctx.flush(stream);
    stream.synchronize();

    // GenAI grows the cache: a new, bigger host tensor holding the old contents in front.
    auto grown = make_cache(5, 100.0f);
    std::memcpy(grown.data(), host.data(), host.get_byte_size());
    const void* grown_dev = ctx.bind(kId, grown, stream);
    const auto dev_after = device_contents(grown_dev, grown.get_size(), stream);
    const auto host_after = host_contents(grown);
    EXPECT_EQ(std::vector<float>(dev_after.begin(), dev_after.begin() + host.get_size()),
              std::vector<float>(host_after.begin(), host_after.begin() + host.get_size()));
}

TEST_F(PagedAttentionContextTest, GrowthWithinCapacityAndShrinkKeepTheBuffer) {
    auto host = make_cache(4, 1.0f);
    const void* dev = ctx.bind(kId, host, stream);
    auto grown = make_cache(5, 1.0f);  // 4 -> 5 blocks: reallocates with headroom (6 blocks)
    const void* dev5 = ctx.bind(kId, grown, stream);
    EXPECT_NE(dev5, dev);
    auto grown6 = make_cache(6, 1.0f);  // fits into the headroom: no reallocation
    EXPECT_EQ(ctx.bind(kId, grown6, stream), dev5);
    auto shrunk = make_cache(2, 1.0f);  // GenAI's clear() + smaller cache: buffer is kept
    EXPECT_EQ(ctx.bind(kId, shrunk, stream), dev5);
    EXPECT_EQ(ctx.bind(kId, grown6, stream), dev5);
}

TEST_F(PagedAttentionContextTest, RefreshAndWriteBackAfterGrowthUseTheNewTensors) {
    auto host = make_cache(2, 1.0f);
    ctx.bind(kId, host, stream);
    auto grown = make_cache(8, 1.0f);
    std::memcpy(grown.data(), host.data(), host.get_byte_size());
    void* dev = ctx.bind(kId, grown, stream);

    // copy-on-write into a block that only exists in the grown cache
    std::memcpy(block(grown, 6), block(grown, 1), kBlockElems * sizeof(float));
    ctx.refresh_blocks(kId, {6}, stream);
    // The kept prefix and the refreshed block match the host (the rest is never-written cache).
    const auto dev_after = device_contents(dev, grown.get_size(), stream);
    const auto host_after = host_contents(grown);
    const auto range = [](const std::vector<float>& v, size_t first_block, size_t num_blocks) {
        return std::vector<float>(v.begin() + first_block * kBlockElems,
                                  v.begin() + (first_block + num_blocks) * kBlockElems);
    };
    EXPECT_EQ(range(dev_after, 0, 2), range(host_after, 0, 2));
    EXPECT_EQ(range(dev_after, 6, 1), range(host_after, 6, 1));

    const std::vector<float> written(kBlockElems, 9.0f);
    stream.upload(CUDA::DevicePointer<void*>{static_cast<float*>(dev) + 7 * kBlockElems},
                  written.data(),
                  written.size() * sizeof(float));
    ctx.write_back(kId, {7});
    ctx.flush(stream);
    stream.synchronize();
    EXPECT_EQ(std::vector<float>(block(grown, 7), block(grown, 7) + kBlockElems), written);
    // the old (smaller) host tensor is no longer written to
    EXPECT_EQ(host_contents(host), host_contents(make_cache(2, 1.0f)));
}

TEST_F(PagedAttentionContextTest, GrowthWhileDesyncedReseedsFromHost) {
    auto host = make_cache(2, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    const std::vector<float> garbage(host.get_size(), -1.0f);  // device diverged from the host
    stream.upload(CUDA::DevicePointer<void*>{dev}, garbage.data(), garbage.size() * sizeof(float));
    ctx.mark_desynced();
    auto grown = make_cache(4, 50.0f);
    std::memcpy(grown.data(), host.data(), host.get_byte_size());
    dev = ctx.bind(kId, grown, stream);
    EXPECT_EQ(device_contents(dev, grown.get_size(), stream), host_contents(grown));
}

TEST_F(PagedAttentionContextTest, RefreshPicksUpHostSideCopyOnWrite) {
    auto host = make_cache(4, 1.0f);
    const void* dev = ctx.bind(kId, host, stream);
    // GenAI's CacheManager::copy_blocks: block 0 -> block 3 on the host only.
    std::memcpy(block(host, 3), block(host, 0), kBlockElems * sizeof(float));
    EXPECT_NE(device_contents(dev, host.get_size(), stream), host_contents(host));

    ctx.refresh_blocks(kId, {3}, stream);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), host_contents(host));
}

TEST_F(PagedAttentionContextTest, DesyncedCacheIsReseededOnNextBind) {
    auto host = make_cache(3, 1.0f);
    void* dev = ctx.bind(kId, host, stream);
    const std::vector<float> written(host.get_size(), -5.0f);
    stream.upload(CUDA::DevicePointer<void*>{dev}, written.data(), written.size() * sizeof(float));
    ctx.write_back(kId, {1});
    // The infer fails before flush(): the pending write-back is dropped...
    ctx.mark_desynced();
    ctx.flush(stream);
    stream.synchronize();
    EXPECT_EQ(host_contents(host), host_contents(make_cache(3, 1.0f)));
    // ...and the mirror is re-seeded from the host on the next bind.
    std::fill_n(host.data<float>(), host.get_size(), 3.0f);
    dev = ctx.bind(kId, host, stream);
    EXPECT_EQ(device_contents(dev, host.get_size(), stream), host_contents(host));
}

TEST_F(PagedAttentionContextTest, UnboundCacheThrows) {
    EXPECT_ANY_THROW(ctx.refresh_blocks(kId, {0}, stream));
    EXPECT_ANY_THROW(ctx.write_back(kId, {0}));
}
