// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <condition_variable>
#include <cuda_executable_network.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_plugin.hpp>
#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/node.hpp>
#include <ops/parameter.hpp>
#include <ops/result.hpp>
#include <random>
#include <threading/ie_executor_manager.hpp>
#include <typeinfo>

#include "nodes/parameter_stub_node.hpp"
#include "nodes/result_stub_node.hpp"
#include "test_networks.hpp"

using namespace InferenceEngine;
using namespace CUDAPlugin;

using devptr_t = DevicePointer<void*>;
using cdevptr_t = DevicePointer<const void*>;

class PluginTest : public testing::Test {
    void SetUp() override { function_ = CreateMatMulTestNetwork(); }

    void TearDown() override {}

public:
    OperationBase::Ptr inOperation;
    OperationBase::Ptr outOperation;
    Blob::Ptr inBlob;
    Blob::Ptr outBlob;
    InferenceEngine::BlobMap blobs;
    std::shared_ptr<ngraph::Function> function_;
};

TEST_F(PluginTest, LoadExecNetwork_Success) {
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_NO_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}}));
}

TEST_F(PluginTest, LoadExecNetwork_NegativeId_Failed) {
    auto dummyFunction = std::make_shared<ngraph::Function>(ov::NodeVector{}, ov::ParameterVector{});
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "-1"}}),
                 InferenceEngine::details::InferenceEngineException);
}

TEST_F(PluginTest, LoadExecNetwork_OutRangeId_Failed) {
    auto dummyFunction = std::make_shared<ngraph::Function>(ov::NodeVector{}, ov::ParameterVector{});
    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{dummyFunction};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(
        plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), std::to_string(CUDA::Device::count())}}),
        InferenceEngine::details::InferenceEngineException);
}

TEST_F(PluginTest, LoadExecNetwork_CudaThreadPool_Success) {
    using namespace std::chrono_literals;

    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}});
    const int deviceId = 0;
    CUDA::Device device{deviceId};
    auto cpuStreamExecutor = std::make_shared<CudaThreadPool>(device, maxConcurrentStreams(device));

    std::unordered_set<std::thread::id> streams;
    std::mutex mtx;
    std::condition_variable condVar;
    cpuStreamExecutor->run([&cpuStreamExecutor, &streams, &mtx, &condVar] {
        auto streamId = std::this_thread::get_id();
        std::this_thread::sleep_for(std::chrono::seconds(2));
        {
            std::lock_guard<std::mutex> lock{mtx};
            streams.insert(streamId);
        }
        condVar.notify_one();
    });
    cpuStreamExecutor->run([&cpuStreamExecutor, &streams, &mtx, &condVar] {
        auto streamId = std::this_thread::get_id();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            std::lock_guard<std::mutex> lock{mtx};
            streams.insert(streamId);
        }
        condVar.notify_one();
    });
    std::unique_lock<std::mutex> lock{mtx};
    condVar.wait_for(lock, 5s, [&streams] { return streams.size() == 2; });
    ASSERT_EQ(streams.size(), 2);
}

TEST_F(PluginTest, LoadExecNetwork_CudaThreadPool_AllJobs_Success) {
    using namespace std::chrono_literals;

    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}});
    const int deviceId = 0;
    CUDA::Device device{deviceId};
    const size_t numConcurrentStreams = maxConcurrentStreams(device);
    auto cpuStreamExecutor = std::make_shared<CudaThreadPool>(device, numConcurrentStreams);

    std::unordered_set<std::thread::id> streams;
    std::unordered_set<const ThreadContext*> threadContexts;
    unsigned numHandledJobs = 0;
    std::mutex mtx;
    std::condition_variable condVar;
    const unsigned numJobs = 2 * numConcurrentStreams;
    for (unsigned i = 0; i < numJobs; ++i) {
        cpuStreamExecutor->run([&cpuStreamExecutor, &threadContexts, &numHandledJobs, &streams, &mtx, &condVar] {
            auto streamId = std::this_thread::get_id();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            {
                std::lock_guard<std::mutex> lock{mtx};
                numHandledJobs += 1;
                streams.insert(streamId);
                threadContexts.insert(&cpuStreamExecutor->GetThreadContext());
            }
            condVar.notify_one();
        });
    }
    std::unique_lock<std::mutex> lock{mtx};
    condVar.wait_for(lock, 10s, [&numHandledJobs, &numJobs] { return numHandledJobs == numJobs; });
    ASSERT_EQ(streams.size(), numConcurrentStreams);
    ASSERT_EQ(threadContexts.size(), numConcurrentStreams);
    ASSERT_EQ(numHandledJobs, numJobs);
}

TEST_F(PluginTest, LoadExecNetwork_CudaThreadPool_AllJobs_Heavy_Success) {
    using namespace std::chrono_literals;

    auto dummyCNNNetwork = InferenceEngine::CNNNetwork{function_};
    Configuration cfg;
    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = plugin->LoadExeNetworkImpl(dummyCNNNetwork, {{CONFIG_KEY(DEVICE_ID), "0"}});
    const int deviceId = 0;
    CUDA::Device device{deviceId};
    const size_t numConcurrentStreams = maxConcurrentStreams(device);
    auto cpuStreamExecutor = std::make_shared<CudaThreadPool>(device, numConcurrentStreams);

    std::unordered_set<std::thread::id> streams;
    std::unordered_set<const ThreadContext*> threadContexts;
    unsigned numHandledJobs = 0;
    std::mutex mtx;
    std::condition_variable condVar;
    const unsigned numJobs = 10 * numConcurrentStreams;
    for (unsigned i = 0; i < numJobs; ++i) {
        cpuStreamExecutor->run([&cpuStreamExecutor, &threadContexts, &numHandledJobs, &streams, &mtx, &condVar] {
            auto streamId = std::this_thread::get_id();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            {
                std::lock_guard<std::mutex> lock{mtx};
                numHandledJobs += 1;
                streams.insert(streamId);
                threadContexts.insert(&cpuStreamExecutor->GetThreadContext());
            }
            condVar.notify_one();
        });
    }
    std::unique_lock<std::mutex> lock{mtx};
    condVar.wait_for(lock, 15s, [&numHandledJobs, &numJobs] { return numHandledJobs == numJobs; });
    ASSERT_EQ(streams.size(), numConcurrentStreams);
    ASSERT_EQ(threadContexts.size(), numConcurrentStreams);
    ASSERT_EQ(numHandledJobs, numJobs);
}
