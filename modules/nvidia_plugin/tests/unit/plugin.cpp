// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <condition_variable>
#include <memory>
#include <ops/parameter.hpp>
#include <ops/result.hpp>
#include <random>
#include <typeinfo>

#include "cuda_compiled_model.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "nodes/parameter_stub_node.hpp"
#include "nodes/result_stub_node.hpp"
#include "test_networks.hpp"

using namespace ov::nvidia_gpu;

using devptr_t = DevicePointer<void*>;
using cdevptr_t = DevicePointer<const void*>;

class PluginTest : public testing::Test {
    void SetUp() override { model_ = create_matmul_test_model(); }

    void TearDown() override {}

public:
    OperationBase::Ptr inOperation;
    OperationBase::Ptr outOperation;
    std::shared_ptr<ov::Model> model_;
};

TEST_F(PluginTest, CompileModel_Success) {
    auto plugin = std::make_shared<Plugin>();
    ASSERT_NO_THROW(plugin->compile_model(model_, {{ov::device::id.name(), "0"}}));
}

TEST_F(PluginTest, CompileModel_NegativeId_Failed) {
    auto dummyFunction = std::make_shared<ov::Model>(ov::OutputVector{}, ov::ParameterVector{});
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->compile_model(dummyFunction, {{ov::device::id.name(), "-1"}}), ov::Exception);
}

TEST_F(PluginTest, CompileModel_OutRangeId_Failed) {
    auto dummyFunction = std::make_shared<ov::Model>(ov::OutputVector{}, ov::ParameterVector{});
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->compile_model(dummyFunction, {{ov::device::id.name(), std::to_string(CUDA::Device::count())}}),
                 ov::Exception);
}

TEST_F(PluginTest, CompileModel_CudaThreadPool_Success) {
    using namespace std::chrono_literals;

    auto plugin = std::make_shared<Plugin>();
    auto execNetwork = plugin->compile_model(model_, {{ov::device::id.name(), "0"}});
    const int deviceId = 0;
    CUDA::Device device{deviceId};
    auto cpuStreamExecutor = std::make_shared<CudaThreadPool>(device, max_concurrent_streams(device));

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

TEST_F(PluginTest, CompileModel_CudaThreadPool_AllJobs_Success) {
    using namespace std::chrono_literals;

    auto plugin = std::make_shared<Plugin>();
    auto compiled_model = plugin->compile_model(model_, {{ov::device::id.name(), "0"}});
    const int deviceId = 0;
    CUDA::Device device{deviceId};
    const size_t numConcurrentStreams = max_concurrent_streams(device);
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
                threadContexts.insert(&cpuStreamExecutor->get_thread_context());
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

TEST_F(PluginTest, CompileModel_CudaThreadPool_AllJobs_Heavy_Success) {
    using namespace std::chrono_literals;

    auto plugin = std::make_shared<Plugin>();
    auto compiled_model = plugin->compile_model(model_, {{ov::device::id.name(), "0"}});
    const int deviceId = 0;
    CUDA::Device device{deviceId};
    const size_t numConcurrentStreams = max_concurrent_streams(device);
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
                threadContexts.insert(&cpuStreamExecutor->get_thread_context());
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
