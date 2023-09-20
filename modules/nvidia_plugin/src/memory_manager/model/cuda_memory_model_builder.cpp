// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_model_builder.hpp"
#include "openvino/core/except.hpp"
#include "memory_manager/model/details/cuda_memory_utils.hpp"

namespace ov {
namespace nvidia_gpu {

void MemoryModelBuilder::addAllocation(BufferID id, int producerIndex, int lastConsumerIndex, size_t bsize) {
    OPENVINO_ASSERT(bsize > 0, "Allocation size is zero!");  // Verify that allocation size isn't zero.
    auto res = offsets_.emplace(id, 0);
    OPENVINO_ASSERT(res.second, "ID is not unique!");  // Verify that "id" is unique.
    const int64_t aligned_size = static_cast<int64_t>(applyAllignment(bsize));
    boxes_.emplace_back(ov::MemorySolver::Box{producerIndex, lastConsumerIndex, aligned_size, id});
}

MemoryModel::Ptr MemoryModelBuilder::build() {
    ov::MemorySolver solver{boxes_};
    const size_t blob_size = solver.solve();
    for (auto& pair : offsets_) pair.second = solver.get_offset(pair.first);

    return std::make_shared<MemoryModel>(blob_size, offsets_);
}

}  // namespace nvidia_gpu
}  // namespace ov
