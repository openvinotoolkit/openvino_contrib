// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/cuda_memory_manager.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "cuda_operation_base.hpp"
#include "memory_manager/cuda_immutable_memory_block_builder.hpp"
#include "memory_manager/model/cuda_memory_model_builder.hpp"

class MemoryManagerTest : public testing::Test, public ov::nvidia_gpu::IOperationMeta {
public:
    using TensorID = ov::nvidia_gpu::TensorID;

    void SetUp() override {
        // Allocate shared memory block for constant tensors
        {
            const std::vector<uint8_t> data(256, 0xA5);
            ov::nvidia_gpu::ImmutableMemoryBlockBuilder builder;
            for (auto id : sharedConstantIds_) {
                builder.addAllocation(id.GetId(), &data[0], data.size());
            }
            std::tie(immutableTensors_, immutableMemoryModel_) = builder.build();
        }

        // Create MemoryModel for mutable tensors
        {
            ov::nvidia_gpu::MemoryModelBuilder builder;
            const size_t size = 1;
            for (int i = 0; i < mutableTensorIDs_.size(); ++i) {
                builder.addAllocation(mutableTensorIDs_[i].GetId(), i, i + 2, size);
            }
            mutableMemoryModel_ = builder.build();
        }
    }

    const std::vector<TensorID> sharedConstantIds_ = {
        TensorID{0}, TensorID{1}, TensorID{2}, TensorID{5}, TensorID{7}, TensorID{9}};
    const std::vector<TensorID> mutableTensorIDs_ = {
        TensorID{101}, TensorID{104}, TensorID{103}, TensorID{105}, TensorID{120}, TensorID{121}};

    std::shared_ptr<ov::nvidia_gpu::DeviceMemBlock> immutableTensors_;
    ov::nvidia_gpu::MemoryModel::Ptr immutableMemoryModel_;
    ov::nvidia_gpu::MemoryModel::Ptr mutableMemoryModel_;

public:  // ov::nvidia_gpu::IOperationMeta
    std::vector<ov::nvidia_gpu::TensorID> inputIds_;
    std::vector<ov::nvidia_gpu::TensorID> outputIds_;
    const std::string& GetName() const override {
        static std::string empty;
        return empty;
    }
    const std::string& GetTypeName() const override { return GetName(); }
    const std::string_view& GetCategory() const override {
        static constexpr std::string_view empty{""};
        return empty;
    }
    const ov::element::Type& GetRuntimePrecision() const override { return ov::element::dynamic; }
    gsl::span<const ov::nvidia_gpu::TensorID> GetInputIds() const override { return inputIds_; }
    gsl::span<const ov::nvidia_gpu::TensorID> GetOutputIds() const override { return outputIds_; }
};

TEST_F(MemoryManagerTest, InputTensorPointersAndTheirOrder) {
    using namespace ov::nvidia_gpu;

    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);

    // Setup operation input identifiers to include all allocated tensors
    inputIds_ = sharedConstantIds_;
    inputIds_.insert(inputIds_.end(), mutableTensorIDs_.begin(), mutableTensorIDs_.end());

    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    // Request device side input pointers providing previously set tensor identifiers
    auto inputTensorPointers = memory_manager->inputTensorPointers(*this, allocation);

    // Verify device side pointers and their order
    EXPECT_TRUE(inputIds_.size() == inputTensorPointers.size());
    const uint8_t* mutableBlockAllocationBase = nullptr;
    for (int i = 0; i < inputIds_.size(); ++i) {
        const TensorID buffer_id = inputIds_[i];
        const void* actual = inputTensorPointers[i].get();
        if (i < sharedConstantIds_.size()) {
            // buffer_id represents tensor from shared constants memory block
            const void* expected = immutableTensors_->deviceTensorPtr(buffer_id);
            EXPECT_EQ(actual, expected);
        } else {
            // buffer_id represents tensor from mutable memory block allocated by MemoryManager
            if (mutableBlockAllocationBase == nullptr) {
                ptrdiff_t offset = -1;
                ASSERT_TRUE(mutableMemoryModel_->offsetForBuffer(buffer_id.GetId(), offset));
                mutableBlockAllocationBase = reinterpret_cast<const uint8_t*>(actual) - offset;
            }

            ptrdiff_t offset = -1;
            ASSERT_TRUE(mutableMemoryModel_->offsetForBuffer(buffer_id.GetId(), offset));
            const void* expected = mutableBlockAllocationBase + offset;
            EXPECT_EQ(actual, expected);
        }
    }
}

TEST_F(MemoryManagerTest, OutputTensorPointersAndTheirOrder) {
    using namespace ov::nvidia_gpu;

    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);

    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    // Request device side output pointers providing previously set tensor identifiers
    auto outputTensorPointers = memory_manager->outputTensorPointers(*this, allocation);

    // Verify device side pointers and their order
    EXPECT_TRUE(outputIds_.size() == outputTensorPointers.size());
    const uint8_t* mutableBlockAllocationBase = nullptr;
    for (int i = 0; i < outputIds_.size(); ++i) {
        const BufferID buffer_id = outputIds_[i].GetId();
        const void* actual = outputTensorPointers[i].get();

        if (mutableBlockAllocationBase == nullptr) {
            ptrdiff_t offset = -1;
            ASSERT_TRUE(mutableMemoryModel_->offsetForBuffer(buffer_id, offset));
            mutableBlockAllocationBase = reinterpret_cast<const uint8_t*>(actual) - offset;
        }

        ptrdiff_t offset = -1;
        ASSERT_TRUE(mutableMemoryModel_->offsetForBuffer(buffer_id, offset));
        const void* expected = mutableBlockAllocationBase + offset;
        EXPECT_EQ(actual, expected);
    }
}

TEST_F(MemoryManagerTest, OperationHasNoInputs) {
    using namespace ov::nvidia_gpu;
    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
    inputIds_ = {};
    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    auto inputTensorPointers = memory_manager->inputTensorPointers(*this, allocation);

    EXPECT_TRUE(inputTensorPointers.empty());
}

TEST_F(MemoryManagerTest, OperationHasNoOutputs) {
    using namespace ov::nvidia_gpu;
    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
    outputIds_ = {};
    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    auto outputTensorPointers = memory_manager->outputTensorPointers(*this, allocation);

    EXPECT_TRUE(outputTensorPointers.empty());
}

TEST_F(MemoryManagerTest, InvalidInputTensorID) {
    using namespace ov::nvidia_gpu;

    const TensorID invalid_buffer_id{9999};
    ASSERT_EQ(0, std::count(sharedConstantIds_.begin(), sharedConstantIds_.end(), invalid_buffer_id));
    ASSERT_EQ(0, std::count(mutableTensorIDs_.begin(), mutableTensorIDs_.end(), invalid_buffer_id));

    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
    inputIds_ = sharedConstantIds_;
    inputIds_.emplace_back(invalid_buffer_id);

    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    ASSERT_THROW(memory_manager->inputTensorPointers(*this, allocation), ov::Exception);
}

TEST_F(MemoryManagerTest, InvalidOutputTensorID) {
    using namespace ov::nvidia_gpu;

    const TensorID invalid_buffer_id{9999};
    ASSERT_EQ(0, std::count(sharedConstantIds_.begin(), sharedConstantIds_.end(), invalid_buffer_id));
    ASSERT_EQ(0, std::count(mutableTensorIDs_.begin(), mutableTensorIDs_.end(), invalid_buffer_id));

    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
    outputIds_ = mutableTensorIDs_;
    outputIds_.emplace_back(invalid_buffer_id);

    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    ASSERT_THROW(memory_manager->outputTensorPointers(*this, allocation), ov::Exception);
}

TEST_F(MemoryManagerTest, ConstantsCanNotBeOutputs) {
    using namespace ov::nvidia_gpu;

    auto memory_manager = std::make_unique<MemoryManager>(immutableTensors_, mutableMemoryModel_);
    outputIds_ = mutableTensorIDs_;
    outputIds_.emplace_back(sharedConstantIds_[0]);

    auto allocation = CUDA::DefaultStream::stream().malloc(immutableTensors_->memoryModel()->deviceMemoryBlockSize() +
                                                           mutableMemoryModel_->deviceMemoryBlockSize());
    ASSERT_THROW(memory_manager->outputTensorPointers(*this, allocation), ov::Exception);
}
