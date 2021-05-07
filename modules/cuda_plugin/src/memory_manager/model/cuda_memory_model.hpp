// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#include <memory>
#include <unordered_map>

namespace CUDAPlugin {

/**
 * @brief MemoryModel describes a size of continous memory blob on CUDA device
 * and a location of every tensor within this blob.
 */
class MemoryModel {
public:
  using Ptr = std::shared_ptr<MemoryModel>;
  using TensorID = unsigned;

  /**
   * @param [in] bsize Memory blob size in bytes.
   * @param [in] offsets Maps tensor identifiers to tensor offsets within a memory blob.
   */
  MemoryModel(size_t bsize, const std::unordered_map<TensorID, ptrdiff_t>& offsets);

  /**
   * @returns The size of memory blob
   */
  size_t deviceMemoryBlobSize() const;

  /**
   * Provides tensor memory offset if any.
   *
   * @param [in] id Tensor identifier.
   * @param [out] offset Tensor memory offset with respect to the beginning of the blob.
   * @returns false if memory blob doesn't contain a tensor with requested identifier.
   */
  bool offsetForTensor(TensorID id, ptrdiff_t& offset) const;

private:
  size_t bsize_;
  std::unordered_map<TensorID, ptrdiff_t> offsets_;
};

}  // namespace CUDAPlugin
