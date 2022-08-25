// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_immutable_memory_model_builder.hpp"

#include <details/ie_exception.hpp>

#include "memory_manager/model/details/cuda_memory_utils.hpp"

namespace ov {
namespace nvidia_gpu {

ImmutableMemoryModelBuilder::ImmutableMemoryModelBuilder() : end_offset_{0} {}

void ImmutableMemoryModelBuilder::addAllocation(BufferID id, size_t bsize) {
    IE_ASSERT(bsize > 0);  // Verify that allocation size isn't zero.
    auto res = offsets_.emplace(id, end_offset_);
    IE_ASSERT(res.second);  // Verify that "id" is unique.
    end_offset_ += applyAllignment(bsize);
}

size_t ImmutableMemoryModelBuilder::deviceMemoryBlockSize() const { return end_offset_; }

MemoryModel::Ptr ImmutableMemoryModelBuilder::build() const {
    return std::make_shared<MemoryModel>(end_offset_, offsets_);
}

}  // namespace nvidia_gpu
}  // namespace ov
