// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_immutable_memory_model_builder.hpp"
#include "openvino/core/except.hpp"
#include "memory_manager/model/details/cuda_memory_utils.hpp"

namespace ov {
namespace nvidia_gpu {

ImmutableMemoryModelBuilder::ImmutableMemoryModelBuilder() : end_offset_{0} {}

void ImmutableMemoryModelBuilder::addAllocation(BufferID id, size_t bsize) {
    OPENVINO_ASSERT(bsize > 0, "Allocation size is zero!");  // Verify that allocation size isn't zero.
    auto res = offsets_.emplace(id, end_offset_);
    OPENVINO_ASSERT(res.second, "ID is not unique!");  // Verify that "id" is unique.
    end_offset_ += applyAllignment(bsize);
}

size_t ImmutableMemoryModelBuilder::deviceMemoryBlockSize() const { return end_offset_; }

MemoryModel::Ptr ImmutableMemoryModelBuilder::build() const {
    return std::make_shared<MemoryModel>(end_offset_, offsets_);
}

}  // namespace nvidia_gpu
}  // namespace ov
