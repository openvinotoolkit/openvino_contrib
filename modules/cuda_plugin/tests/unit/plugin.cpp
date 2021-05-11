// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <typeinfo>
#include <condition_variable>
#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <cuda_plugin.hpp>
#include <cuda_executable_network.hpp>
#include <ops/saxpy_op.hpp>
#include <cuda/device.hpp>
#include <threading/ie_executor_manager.hpp>

using namespace CUDAPlugin;

class PluginTest : public testing::Test {
    void SetUp() override {
    }

    void TearDown() override {
    }

 public:
    std::shared_ptr<InferenceEngine::CPUStreamsExecutor>
    GetCpuStreamExecutor(const int deviceId, cudaDeviceProp deviceProp) {
        const size_t numConcurrentStreams = CudaDevice::GetDeviceConcurrentKernels(deviceProp);
        InferenceEngine::IStreamsExecutor::Config cudaExecutorConfig(
            "CudaGpuExecutor:Device=" + std::to_string(deviceId),
            numConcurrentStreams);
        auto streamExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(cudaExecutorConfig);
        return std::dynamic_pointer_cast<InferenceEngine::CPUStreamsExecutor>(streamExecutor);
    }
};

TEST_F(PluginTest, LoadExecNetwork_Success) {
    auto dummyFunction = std::make_shared<ngraph::Function>(
        ngraph::NodeVector{}, ngraph::ParameterVector{});
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}}));
}

TEST_F(PluginTest, LoadExecNetwork_NegativeId_Failed) {
    auto dummyFunction = std::make_shared<ngraph::Function>(
        ngraph::NodeVector{}, ngraph::ParameterVector{});
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "-1"}}),
                 InferenceEngine::details::InferenceEngineException);
}

TEST_F(PluginTest, LoadExecNetwork_OutRangeId_Failed) {
    auto dummyFunction = std::make_shared<ngraph::Function>(
        ngraph::NodeVector{}, ngraph::ParameterVector{});
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto devicesProp = CudaDevice::GetAllDevicesProp();
    ASSERT_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), std::to_string(devicesProp.size())}}),
                 InferenceEngine::details::InferenceEngineException);
}

TEST_F(PluginTest, LoadExecNetwork_CudaThreadPool_Success) {
    using namespace std::chrono_literals;

    auto dummyFunction = std::make_shared<ngraph::Function>(
        ngraph::NodeVector{}, ngraph::ParameterVector{});
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}});
    auto devicesProp = CudaDevice::GetAllDevicesProp();
    ASSERT_TRUE(!devicesProp.empty());
    const int deviceId = 0;
    auto cpuStreamExecutor = GetCpuStreamExecutor(deviceId, devicesProp[deviceId]);

    std::unordered_set<int> streams;
    std::mutex mtx;
    std::condition_variable condVar;
    cpuStreamExecutor->run([&cpuStreamExecutor, &streams, &mtx, &condVar] {
        auto streamId = cpuStreamExecutor->GetStreamId();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        {
            std::unique_lock<std::mutex> lock{mtx};
            streams.insert(streamId);
        }
        condVar.notify_one();
    });
    cpuStreamExecutor->run([&cpuStreamExecutor, &streams, &mtx, &condVar] {
        auto streamId = cpuStreamExecutor->GetStreamId();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            std::unique_lock<std::mutex> lock{mtx};
            streams.insert(streamId);
        }
        condVar.notify_one();
    });
    std::unique_lock<std::mutex> lock{mtx};
    condVar.wait_for(lock, 5s, [&streams] {
        return streams.size() == 2;
    });
    ASSERT_EQ(streams.size(), 2);
}
