// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_immutable_memory_model_builder.hpp"

#include <details/ie_exception.hpp>

#include "memory_manager/model/details/cuda_memory_utils.hpp"

namespace CUDAPlugin {

ImmutableMemoryModelBuilder::ImmutableMemoryModelBuilder()
  : end_offset_{ 0 }
{
}

void ImmutableMemoryModelBuilder::addAllocation(TensorID id, size_t bsize) {
  IE_ASSERT(bsize > 0); // Verify that allocation size isn't zero.
  auto res = offsets_.emplace(id, end_offset_);
  IE_ASSERT(res.second); // Verify that "id" is unique.
  end_offset_ += applyAllignment(bsize);
}

MemoryModel::Ptr ImmutableMemoryModelBuilder::build() {
  return std::make_shared<MemoryModel>(end_offset_, offsets_);
}

}  // namespace CUDAPlugin
