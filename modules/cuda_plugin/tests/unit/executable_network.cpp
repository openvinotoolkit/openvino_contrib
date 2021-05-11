// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <typeinfo>
#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <cuda_plugin.hpp>
#include <cuda_executable_network.hpp>
#include <ops/saxpy_op.hpp>

using namespace CUDAPlugin;

class ExecNetworkTest : public testing::Test {
    void SetUp() override {
    }

    void TearDown() override {
    }

 public:
    const auto& GetExecSequence(std::shared_ptr<ExecutableNetwork>& execNetwork) {
        return execNetwork->exec_sequence_;
    }
};

class SaxpyDummyNode : public ngraph::op::Sink {
 public:
    inline static constexpr type_info_t type_info{"Saxpy", 0};
    const type_info_t& get_type_info() const override {
        return type_info;
    }

    std::shared_ptr<ngraph::Node>
    clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        return std::make_shared<SaxpyDummyNode>();
    }
};

class SuperOperationDummyNode : public ngraph::op::Sink {
 public:
    inline static constexpr type_info_t type_info{"SuperOperation", 0};
    const type_info_t& get_type_info() const override {
        return type_info;
    }

    std::shared_ptr<ngraph::Node>
    clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        return std::make_shared<SuperOperationDummyNode>();
    }
};

TEST_F(ExecNetworkTest, BuildExecutableSequence_Saxpy_Success) {
    auto dummyFunction = std::make_shared<ngraph::Function>(
        ngraph::NodeVector{}, ngraph::ParameterVector{});
    auto saxpyDummyNode = std::make_shared<SaxpyDummyNode>();
    dummyFunction->add_sinks({ saxpyDummyNode });
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = std::dynamic_pointer_cast<ExecutableNetwork>(
        plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}}));
    const auto& execSequence = GetExecSequence(execNetwork);
    ASSERT_EQ(execSequence.size(), 1);
    ASSERT_EQ(std::type_index(typeid(*execSequence[0].get())), std::type_index(typeid(SaxpyOp)));
}

TEST_F(ExecNetworkTest, BuildExecutableSequence_SuperOperation_Failed) {
    auto dummyFunction = std::make_shared<ngraph::Function>(
        ngraph::NodeVector{}, ngraph::ParameterVector{});
    auto superOperationDummyNode = std::make_shared<SuperOperationDummyNode>();
    dummyFunction->add_sinks({ superOperationDummyNode });
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}}),
                 InferenceEngine::details::InferenceEngineException);
}