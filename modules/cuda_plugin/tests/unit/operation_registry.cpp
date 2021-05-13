// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <typeinfo>
#include <gtest/gtest.h>

#include <ngraph/node.hpp>
#include <cuda_operation_registry.hpp>
#include <ops/saxpy_op.hpp>

using namespace CUDAPlugin;

class OperationRegistryTest : public testing::Test {
  void SetUp() override {
  }

  void TearDown() override {
  }

 public:
    std::vector<unsigned> dummyInputTensorIds = std::vector<unsigned>{1, 2, 3};
    std::vector<unsigned> dummyOutputTensorIds = std::vector<unsigned>{4, 5, 6};
};

class SaxpyDummyNode : public ngraph::Node {
 public:
  static constexpr type_info_t type_info{"Saxpy", 0};
  const type_info_t& get_type_info() const override {
    return type_info;
  }

  std::shared_ptr<ngraph::Node>
  clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
    return std::make_shared<SaxpyDummyNode>();
  }
};
constexpr ngraph::Node::type_info_t SaxpyDummyNode::type_info;

class SuperOperationDummyNode : public ngraph::Node {
 public:
    static constexpr type_info_t type_info{"SuperOperation", 0};
    const type_info_t& get_type_info() const override {
        return type_info;
    }

    std::shared_ptr<ngraph::Node>
    clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        return std::make_shared<SuperOperationDummyNode>();
    }
};
constexpr ngraph::Node::type_info_t SuperOperationDummyNode::type_info;

TEST_F(OperationRegistryTest, CheckOperation_Available) {
    auto saxpyDummyNode = std::make_shared<SaxpyDummyNode>();
    ASSERT_TRUE(OperationRegistry::getInstance().hasOperation(saxpyDummyNode));
}

TEST_F(OperationRegistryTest, CheckOperation_NotFound) {
    auto superOperationDummyNode = std::make_shared<SuperOperationDummyNode>();
    ASSERT_FALSE(OperationRegistry::getInstance().hasOperation(superOperationDummyNode));
}

TEST_F(OperationRegistryTest, GetOperationTypename) {
    auto saxpyDummyNode = std::make_shared<SaxpyDummyNode>();
    ASSERT_EQ(OperationRegistry::getInstance().getOperationTypename(saxpyDummyNode), "SaxpyOp");
}

TEST_F(OperationRegistryTest, GetOperationBuilder_Available) {
    auto saxpyDummyNode = std::make_shared<SaxpyDummyNode>();
    ASSERT_TRUE(OperationRegistry::getInstance().createOperation(saxpyDummyNode, dummyInputTensorIds, dummyOutputTensorIds));
}

TEST_F(OperationRegistryTest, GetOperationBuilder_NotFound) {
    auto superOperationDummyNode = std::make_shared<SuperOperationDummyNode>();
    ASSERT_THROW(OperationRegistry::getInstance().createOperation(superOperationDummyNode, dummyInputTensorIds, dummyOutputTensorIds), std::out_of_range);
}

TEST_F(OperationRegistryTest, BuildOperation_Saxpy) {
    auto saxpyDummyNode = std::make_shared<SaxpyDummyNode>();
    auto saxpyOperation = OperationRegistry::getInstance().createOperation(saxpyDummyNode, dummyInputTensorIds, dummyOutputTensorIds);
    ASSERT_EQ(std::type_index(typeid(*saxpyOperation.get())), std::type_index(typeid(SaxpyOp)));
}

TEST_F(OperationRegistryTest, BuildOperationAndCheckTenorIds_Success) {
    auto saxpyDummyNode = std::make_shared<SaxpyDummyNode>();
    auto saxpyDummyOperation = OperationRegistry::getInstance().createOperation(saxpyDummyNode, dummyInputTensorIds, dummyOutputTensorIds);
    ASSERT_TRUE(saxpyDummyOperation);
    auto inputIds = saxpyDummyOperation->GetInputIds();
    auto outputIds = saxpyDummyOperation->GetOutputIds();
    ASSERT_EQ(std::vector<unsigned>(inputIds.begin(), inputIds.end()), dummyInputTensorIds);
    ASSERT_EQ(std::vector<unsigned>(outputIds.begin(), outputIds.end()), dummyOutputTensorIds);
}
