// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_model_builder.hpp"

#include <details/ie_exception.hpp>

#include "memory_manager/model/details/cuda_memory_utils.hpp"

namespace CUDAPlugin {

void MemoryModelBuilder::addAllocation(TensorID id, int start, int end, size_t bsize) {
  const int64_t aligned_size = static_cast<int64_t>(applyAllignment(bsize));
  boxes_.emplace_back(MemorySolver::Box{ start, end, aligned_size, id });
  auto res = offsets_.emplace(id, 0);
  IE_ASSERT(res.second); // Verify that "id" is unique.
}

MemoryModel::Ptr MemoryModelBuilder::build() {
  MemorySolver solver {boxes_};
  const size_t blob_size = solver.solve();
  for (auto& pair : offsets_)
    pair.second = solver.getOffset(pair.first);

  return std::make_shared<MemoryModel>(blob_size, offsets_);
}

}  // namespace CUDAPlugin
