// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cuda/cuda_config.hpp>
#include <cuda_executable_network.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_plugin.hpp>
#include <functional_test_utils/include/functional_test_utils/precision_utils.hpp>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <ops/matmul.hpp>
#include <typeinfo>

#include "test_networks.hpp"

using namespace CUDAPlugin;

class ExecNetworkTest : public testing::Test {
    void SetUp() override {
        function_ = CreateMatMulTestNetwork();
        super_function_ = CreateSuperOperationTestNetwork();
    }

    void TearDown() override {}

public:
    auto GetExecSequence(const std::shared_ptr<ExecutableNetwork>& execNetwork) {
        const auto& graph = *execNetwork->graph_;
        std::vector<OperationBase::Ptr> execSequence{};
        execSequence.insert(execSequence.end(), graph.exec_sequence_.begin(), graph.exec_sequence_.end());
        return execSequence;
    }
    const auto& GetMemoryManagerPool(const std::shared_ptr<ExecutableNetwork>& execNetwork) {
        return execNetwork->memory_pool_;
    }

    std::shared_ptr<ngraph::Function> function_;
    std::shared_ptr<ngraph::Function> super_function_;
};

TEST_F(ExecNetworkTest, BuildExecutableSequence_MatMul_Success) {
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = std::dynamic_pointer_cast<ExecutableNetwork>(
        plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}}));
    const auto& execSequence = GetExecSequence(execNetwork);
    ASSERT_EQ(execSequence.size(), 3);
    ASSERT_EQ(std::type_index(typeid(*execSequence[1].get())), std::type_index(typeid(MatMulOp)));
}

TEST_F(ExecNetworkTest, BuildExecutableSequence_SuperOperation_Failed) {
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{super_function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}}),
                 InferenceEngine::details::InferenceEngineException);
}

TEST_F(ExecNetworkTest, LoadExecNetwork_OptimalNumberInferRequests_1_Success) {
    using namespace std::chrono_literals;

    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 1;
    auto execNetwork =
        plugin->LoadExeNetworkImpl(dummyCNNNetwork,
                                   {
                                       {CONFIG_KEY(DEVICE_ID), "0"},
                                       {CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), std::to_string(total_streams)},
                                   });
    auto cudaExecNetwork = std::dynamic_pointer_cast<ExecutableNetwork>(execNetwork);
    auto& memoryManagerPool = GetMemoryManagerPool(cudaExecNetwork);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
}

TEST_F(ExecNetworkTest, LoadExecNetwork_OptimalNumberInferRequests_8_Success) {
    using namespace std::chrono_literals;

    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 8;
    auto execNetwork =
        plugin->LoadExeNetworkImpl(dummyCNNNetwork,
                                   {
                                       {CONFIG_KEY(DEVICE_ID), "0"},
                                       {CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), std::to_string(total_streams)},
                                   });
    auto cudaExecNetwork = std::dynamic_pointer_cast<ExecutableNetwork>(execNetwork);
    auto& memoryManagerPool = GetMemoryManagerPool(cudaExecNetwork);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
}

TEST_F(ExecNetworkTest, LoadExecNetwork_OptimalNumberInferRequests_Auto_Success) {
    using namespace std::chrono_literals;

    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork =
        plugin->LoadExeNetworkImpl(dummyCNNNetwork,
                                   {
                                       {CONFIG_KEY(DEVICE_ID), "0"},
                                       {CUDA_CONFIG_KEY(THROUGHPUT_STREAMS), CUDA_CONFIG_VALUE(THROUGHPUT_AUTO)},
                                   });
    auto cudaExecNetwork = std::dynamic_pointer_cast<ExecutableNetwork>(execNetwork);
    auto& memoryManagerPool = GetMemoryManagerPool(cudaExecNetwork);
    ASSERT_GT(memoryManagerPool->Size(), 1);
}
