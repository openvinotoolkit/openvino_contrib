// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/cuda_immutable_memory_block_builder.hpp"
#include "memory_manager/model/cuda_memory_model_builder.hpp"

#include "cuda_operation_base.hpp"

#include <memory>
#include <vector>

class MemoryManagerTest : public testing::Test,
                          public CUDAPlugin::IOperationMeta {
public:
  void SetUp() override {
    // Allocate shared memory block for constant tensors
    {
      const std::vector<uint8_t> data(256, 0xA5);
      CUDAPlugin::ImmutableMemoryBlockBuilder builder;
      for (auto id : sharedConstantIds_) {
        builder.addAllocation(id, &data[0], data.size());
      }
      immutableTensors_ = builder.build();
    }

    // Create MemoryModel for mutable tensors
    {
      CUDAPlugin::MemoryModelBuilder builder;
      const size_t size = 1;
      for (int i = 0; i< mutableTensorIds_.size(); ++i) {
        builder.addAllocation(mutableTensorIds_[i], i, i + 2, size);
      }
      mutableMemoryModel_ = builder.build();
    }
  }

  const std::vector<unsigned> sharedConstantIds_ = {
    0, 1, 2, 5, 7, 9
  };
  const std::vector<unsigned> mutableTensorIds_ = {
    101, 104, 103, 105, 120, 121
  };

  std::shared_ptr<CUDAPlugin::DeviceMemBlock> immutableTensors_;
  CUDAPlugin::MemoryModel::Ptr mutableMemoryModel_;

public: // CUDAPlugin::IOperationMeta
  std::vector<unsigned> inputIds_;
  std::vector<unsigned> outputIds_;
  const std::string& GetName() const override {
    static std::string empty;
    return empty;
  }
  gsl::span<const unsigned> GetInputIds() const override { return inputIds_; }
  gsl::span<const unsigned> GetOutputIds() const override { return outputIds_; }
};

TEST_F(MemoryManagerTest, InputTensorPointersAndTheirOrder) {
  using namespace CUDAPlugin;

  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);

  // Setup operation input identifiers to include all allocated tensors
  inputIds_ = sharedConstantIds_;
  inputIds_.insert(inputIds_.end(), mutableTensorIds_.begin(), mutableTensorIds_.end());

  // Request device side input pointers providing previously set tensor identifiers
  auto inputTensorPointers = memory_manager->inputTensorPointers(*this);

  // Verify device side pointers and their order
  EXPECT_TRUE(inputIds_.size() == inputTensorPointers.size());
  const uint8_t* mutableBlockAllocationBase = nullptr;
  for (int i = 0; i < inputIds_.size(); ++i) {
    const MemoryModel::TensorID tensor_id = inputIds_[i];
    const void* actual = inputTensorPointers[i].get();
    if (i < sharedConstantIds_.size()) {
      // tensor_id represents tensor from shared constants memory block
      const void* expected = immutableTensors_->deviceTensorPtr(tensor_id);
      EXPECT_EQ(actual, expected);
    } else {
      // tensor_id represents tensor from mutable memory block allocated by MemoryManager
      if (mutableBlockAllocationBase == nullptr) {
        ptrdiff_t offset = -1;
        ASSERT_TRUE(mutableMemoryModel_->offsetForTensor(tensor_id, offset));
        mutableBlockAllocationBase = reinterpret_cast<const uint8_t*>(actual) - offset;
      }

      ptrdiff_t offset = -1;
      ASSERT_TRUE(mutableMemoryModel_->offsetForTensor(tensor_id, offset));
      const void* expected = mutableBlockAllocationBase + offset;
      EXPECT_EQ(actual, expected);
    }
  }
}

TEST_F(MemoryManagerTest, OutputTensorPointersAndTheirOrder) {
  using namespace CUDAPlugin;

  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);

  // Setup operation output identifiers to include tensors from mutable memory blob only.
  outputIds_ = mutableTensorIds_;

  // Request device side output pointers providing previously set tensor identifiers
  auto outputTensorPointers = memory_manager->outputTensorPointers(*this);

  // Verify device side pointers and their order
  EXPECT_TRUE(outputIds_.size() == outputTensorPointers.size());
  const uint8_t* mutableBlockAllocationBase = nullptr;
  for (int i = 0; i < outputIds_.size(); ++i) {
    const MemoryModel::TensorID tensor_id = outputIds_[i];
    const void* actual = outputTensorPointers[i].get();

    if (mutableBlockAllocationBase == nullptr) {
      ptrdiff_t offset = -1;
      ASSERT_TRUE(mutableMemoryModel_->offsetForTensor(tensor_id, offset));
      mutableBlockAllocationBase = reinterpret_cast<const uint8_t*>(actual) - offset;
    }

    ptrdiff_t offset = -1;
    ASSERT_TRUE(mutableMemoryModel_->offsetForTensor(tensor_id, offset));
    const void* expected = mutableBlockAllocationBase + offset;
    EXPECT_EQ(actual, expected);
  }
}

TEST_F(MemoryManagerTest, OperationHasNoInputs) {
  using namespace CUDAPlugin;
  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
  inputIds_ = {};
  auto inputTensorPointers = memory_manager->inputTensorPointers(*this);

  EXPECT_TRUE(inputTensorPointers.empty());
}

TEST_F(MemoryManagerTest, OperationHasNoOutputs) {
  using namespace CUDAPlugin;
  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
  outputIds_ = {};
  auto outputTensorPointers = memory_manager->outputTensorPointers(*this);

  EXPECT_TRUE(outputTensorPointers.empty());
}

TEST_F(MemoryManagerTest, InvalidInputTensorID) {
  using namespace CUDAPlugin;

  const MemoryModel::TensorID invalid_tensor_id = 9999;
  ASSERT_EQ(0, std::count(sharedConstantIds_.begin(), sharedConstantIds_.end(), invalid_tensor_id));
  ASSERT_EQ(0, std::count(mutableTensorIds_.begin(), mutableTensorIds_.end(), invalid_tensor_id));

  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
  inputIds_ = sharedConstantIds_;
  inputIds_.emplace_back(invalid_tensor_id);

  #ifdef NDEBUG
    ASSERT_THROW(memory_manager->inputTensorPointers(*this), InferenceEngine::details::InferenceEngineException);
  #else
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH(memory_manager->inputTensorPointers(*this), "Assertion");
  #endif
}

TEST_F(MemoryManagerTest, InvalidOutputTensorID) {
  using namespace CUDAPlugin;

  const MemoryModel::TensorID invalid_tensor_id = 9999;
  ASSERT_EQ(0, std::count(sharedConstantIds_.begin(), sharedConstantIds_.end(), invalid_tensor_id));
  ASSERT_EQ(0, std::count(mutableTensorIds_.begin(), mutableTensorIds_.end(), invalid_tensor_id));

  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
  outputIds_ = mutableTensorIds_;
  outputIds_.emplace_back(invalid_tensor_id);

  #ifdef NDEBUG
    ASSERT_THROW(memory_manager->outputTensorPointers(*this), InferenceEngine::details::InferenceEngineException);
  #else
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH(memory_manager->outputTensorPointers(*this), "Assertion");
  #endif
}

TEST_F(MemoryManagerTest, ConstantsCanNotBeOutputs) {
  using namespace CUDAPlugin;

  auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
  outputIds_ = mutableTensorIds_;
  outputIds_.emplace_back(sharedConstantIds_[0]);

  #ifdef NDEBUG
    ASSERT_THROW(memory_manager->outputTensorPointers(*this), InferenceEngine::details::InferenceEngineException);
  #else
    testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_DEATH(memory_manager->outputTensorPointers(*this), "Assertion");
  #endif
}
