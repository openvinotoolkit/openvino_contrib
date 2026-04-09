// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_memory_model_builder.hpp"

#include <gtest/gtest.h>

#include "memory_manager/model/details/cuda_memory_utils.hpp"
#include "openvino/core/except.hpp"

/*
  Let's assume we have 5 nodes: parameter, result and 3 operations
  in between:
          Op: | ParameterOp0 | Op1 | Op2 | Op3 | ResultOp4
    Op index: |     0        |  1  |  2  |  3  |    4

  5 nodes are connected by 4 tensors (t0...t3) so they form an execution
  sequence passing a data from a previous node to the next:
                    t0       t1       t2       t3
      ParameterOp0 ---> Op1 ---> Op2 ---> Op3 ---> ResultOp4

  ParameterOp0 and ResultOp4 are copying tensor data to device and
  to host respectively.

  For such scenario tensor lifetimes are:

    tensors
       |
    t0 | ========   .    .    .
    t1 |  .   ========   .    .
    t2 |  .    .   ========   .
    t3 |  .    .    .   =======
       |__.____.____.____.____.__ op indices
          0    1    2    3    4

  Expected allocations for this case would be:

     Mem(offset)
      |
      |      ____  ____
      |     | t1 || t3 |
      |   __|____||____|
      |  | t0 || t2 |
      |__|____||____|_____ op indices
          0  1  2  3  4
  OR
     Mem(offset)
      |
      |   ____  ____
      |  | t0 || t2 |
      |  |____||____|__
      |     | t1 || t3 |
      |_____|____||____|__ op indices
          0  1  2  3  4

  Note: this test replicates MemSolverTest.GetOffset test.
  It's intended to verify that MemoryModelBuilder is using
  MemorySolver correctly.
 */
TEST(MemoryModelBuilder, Build) {
    using namespace ov::nvidia_gpu;

    MemoryModelBuilder builder;
    const size_t size = 1;
    const size_t allocation_size = applyAllignment(size);
    builder.addAllocation(0, 0, 1, size);
    builder.addAllocation(1, 1, 2, size);
    builder.addAllocation(2, 2, 3, size);
    builder.addAllocation(3, 3, 4, size);

    MemoryModel::Ptr model = builder.build();
    auto existingOffset = [&model](BufferID id, ptrdiff_t& offset) { ASSERT_TRUE(model->offsetForBuffer(id, offset)); };
    auto offsetForId = [&existingOffset](BufferID id) {
        ptrdiff_t offset = -1;
        existingOffset(id, offset);
        return offset;
    };

    // Expected offsets are:
    //  - [t0: 0,               t1: allocation_size, t2: 0,               t3: allocation_size]
    // or
    //  - [t0: allocation_size, t1: 0,               t2: allocation_size, t3: 0]
    EXPECT_EQ(offsetForId(0) + offsetForId(1), allocation_size);
    EXPECT_EQ(offsetForId(1) + offsetForId(2), allocation_size);
    EXPECT_EQ(offsetForId(2) + offsetForId(3), allocation_size);

    // Expect 2 allocations at a time at max since each op consumes
    // 1 input and produces 1 output
    EXPECT_EQ(model->deviceMemoryBlockSize(), 2 * allocation_size);
}

TEST(MemoryModelBuilder, HandleDuplicateAllocation) {
    using namespace ov::nvidia_gpu;

    MemoryModelBuilder builder;

    BufferID duplicate_buffer_id = 1;
    size_t size1 = 128;
    size_t size2 = 256;

    builder.addAllocation(duplicate_buffer_id, 0, 1, size1);

    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, 0, 1, size1), ov::Exception);
    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, 0, 1, size2), ov::Exception);
    ASSERT_THROW(builder.addAllocation(duplicate_buffer_id, 1, 2, size1), ov::Exception);
}

TEST(MemoryModelBuilder, HandleZeroAllocationSize) {
    using namespace ov::nvidia_gpu;

    MemoryModelBuilder builder;

    BufferID buffer_id = 1;
    const size_t size = 0;

    ASSERT_THROW(builder.addAllocation(buffer_id, 0, 1, size), ov::Exception);
}

/*
 * Impossible edge case.
 * Just to clarify class behaviour.
 */
TEST(MemoryModelBuilder, NoAllocations) {
    using namespace ov::nvidia_gpu;

    MemoryModelBuilder builder;
    MemoryModel::Ptr model = builder.build();

    EXPECT_EQ(model->deviceMemoryBlockSize(), 0);
}
