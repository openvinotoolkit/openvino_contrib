// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "memory_manager/model/cuda_immutable_memory_model_builder.hpp"
#include "memory_manager/model/details/cuda_memory_utils.hpp"

#include <details/ie_exception.hpp>


TEST(ImmutableMemoryModelBuilder, BuildEmpty) {
  using namespace CUDAPlugin;

  ImmutableMemoryModelBuilder builder;
  MemoryModel::Ptr model = builder.build();
  ASSERT_EQ(model->deviceMemoryBlockSize(), 0);
}

TEST(ImmutableMemoryModelBuilder, Build) {
  using namespace CUDAPlugin;

  BufferID id1 = 0;
  size_t size1 = 1;
  BufferID id2 = 3;
  size_t size2 = 256;
  BufferID id3 = 7;
  size_t size3 = 784;
  BufferID id4 = 9;
  size_t size4 = 1396;

  ImmutableMemoryModelBuilder builder;
  builder.addAllocation(id1, size1);
  builder.addAllocation(id2, size2);
  builder.addAllocation(id3, size3);
  builder.addAllocation(id4, size4);
  MemoryModel::Ptr model = builder.build();

  ptrdiff_t offset1 = -1;
  ASSERT_TRUE(model->offsetForTensor(id1, offset1));
  ASSERT_EQ(offset1, 0);

  ptrdiff_t offset2 = -1;
  ASSERT_TRUE(model->offsetForTensor(id2, offset2));
  ASSERT_EQ(offset2, offset1 + applyAllignment(size1));

  ptrdiff_t offset3 = -1;
  ASSERT_TRUE(model->offsetForTensor(id3, offset3));
  ASSERT_EQ(offset3, offset2 + applyAllignment(size2));

  ptrdiff_t offset4 = -1;
  ASSERT_TRUE(model->offsetForTensor(id4, offset4));
  ASSERT_EQ(offset4, offset3 + applyAllignment(size3));

  ASSERT_EQ(model->deviceMemoryBlockSize(), offset4 + applyAllignment(size4));
}

TEST(ImmutableMemoryModelBuilder, HandleDuplicateAllocation) {
  using namespace CUDAPlugin;

  ImmutableMemoryModelBuilder builder;

  BufferID duplicate_buffer_id = 1;
  size_t size1 = 128;
  size_t size2 = 256;

  builder.addAllocation(duplicate_buffer_id, size1);

  #ifdef NDEBUG
    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, size1), InferenceEngine::details::InferenceEngineException);
    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, size2), InferenceEngine::details::InferenceEngineException);
  #else
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH(builder.addAllocation(duplicate_buffer_id, size1), "Assertion");
    ASSERT_DEATH(builder.addAllocation(duplicate_buffer_id, size2), "Assertion");
  #endif
}

TEST(ImmutableMemoryModelBuilder, HandleZeroAllocationSize) {
  using namespace CUDAPlugin;

  ImmutableMemoryModelBuilder builder;

  BufferID buffer_id = 1;

  #ifdef NDEBUG
    ASSERT_THROW(builder.addAllocation(buffer_id, 0), InferenceEngine::details::InferenceEngineException);
  #else
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH(builder.addAllocation(buffer_id, 0), "Assertion");
  #endif
}
