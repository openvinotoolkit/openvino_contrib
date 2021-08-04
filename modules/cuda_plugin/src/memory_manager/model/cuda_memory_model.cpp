// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_memory_model.hpp"

namespace CUDAPlugin {

MemoryModel::MemoryModel(size_t bsize, const std::unordered_map<BufferID, ptrdiff_t>& offsets)
  : bsize_{ bsize }, offsets_{ offsets }
{
}

size_t MemoryModel::deviceMemoryBlockSize() const {
  return bsize_;
}

bool MemoryModel::offsetForTensor(BufferID id, ptrdiff_t& offset) const {
  auto it = offsets_.find(id);
  if (it == offsets_.end())
    return false;
  offset = it->second;
  return true;
}

}  // namespace CUDAPlugin
