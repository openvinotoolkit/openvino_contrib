// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda_operation_registry.hpp>
#include <memory>
#include <ngraph/node.hpp>
#include <ops/parameter.hpp>
#include <typeinfo>
#include <vector>

using namespace ov::nvidia_gpu;

class OperationRegistryTest : public testing::Test {
    void SetUp() override {}

    void TearDown() override {}

public:
    std::vector<TensorID> dummyInputBufferIds = {TensorID{1}, TensorID{2}, TensorID{3}};
    std::vector<TensorID> dummyOutputBufferIds = {TensorID{4}, TensorID{5}, TensorID{6}};
    CUDA::Device device_{};
    bool optimizeOption = false;
};

class ParameterDummyNode : public ov::Node {
public:
    static constexpr type_info_t type_info{"Parameter", 0ul};
    const type_info_t& get_type_info() const override { return type_info; }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<ParameterDummyNode>();
    }
};
constexpr ov::Node::type_info_t ParameterDummyNode::type_info;

class SuperOperationDummyNode : public ov::Node {
public:
    static constexpr type_info_t type_info{"SuperOperation", 0ul};
    const type_info_t& get_type_info() const override { return type_info; }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<SuperOperationDummyNode>();
    }
};
constexpr ov::Node::type_info_t SuperOperationDummyNode::type_info;

TEST_F(OperationRegistryTest, CheckOperation_Available) {
    auto parameterDummyNode = std::make_shared<ParameterDummyNode>();
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(parameterDummyNode));
}

TEST_F(OperationRegistryTest, CheckOperation_NotFound) {
    auto superOperationDummyNode = std::make_shared<SuperOperationDummyNode>();
    ASSERT_FALSE(OperationRegistry::getInstance().hasOperation(superOperationDummyNode));
}

TEST_F(OperationRegistryTest, GetOperationBuilder_Available) {
    auto parameterDummyNode = std::make_shared<ParameterDummyNode>();
    ASSERT_TRUE(OperationRegistry::getInstance().createOperation(
        CreationContext{device_, optimizeOption}, parameterDummyNode, dummyInputBufferIds, dummyOutputBufferIds));
}

TEST_F(OperationRegistryTest, GetOperationBuilder_NotFound) {
    auto superOperationDummyNode = std::make_shared<SuperOperationDummyNode>();
    ASSERT_THROW(OperationRegistry::getInstance().createOperation(CreationContext{device_, optimizeOption},
                                                                  superOperationDummyNode,
                                                                  dummyInputBufferIds,
                                                                  dummyOutputBufferIds),
                 std::out_of_range);
}

TEST_F(OperationRegistryTest, BuildOperation_Parameter) {
    auto parameterDummyNode = std::make_shared<ParameterDummyNode>();
    auto parameterOperation = OperationRegistry::getInstance().createOperation(
        CreationContext{device_, optimizeOption}, parameterDummyNode, dummyInputBufferIds, dummyOutputBufferIds);
    ASSERT_EQ(std::type_index(typeid(*parameterOperation.get())), std::type_index(typeid(ParameterOp)));
}

TEST_F(OperationRegistryTest, BuildOperationAndCheckTenorIds_Success) {
    auto parameterDummyNode = std::make_shared<ParameterDummyNode>();
    auto parameterOperation = OperationRegistry::getInstance().createOperation(
        CreationContext{device_, optimizeOption}, parameterDummyNode, dummyInputBufferIds, dummyOutputBufferIds);
    ASSERT_TRUE(parameterOperation);
    auto inputIds = parameterOperation->GetInputIds();
    auto outputIds = parameterOperation->GetOutputIds();
    ASSERT_EQ(std::vector<TensorID>(inputIds.begin(), inputIds.end()), dummyInputBufferIds);
    ASSERT_EQ(std::vector<TensorID>(outputIds.begin(), outputIds.end()), dummyOutputBufferIds);
}
