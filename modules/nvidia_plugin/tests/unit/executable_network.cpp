// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <typeinfo>

#include "cuda_compiled_model.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"

#include <ngraph/node.hpp>
#include <nvidia/nvidia_config.hpp>
#include <ops/matmul.hpp>

#include "test_networks.hpp"

using namespace ov::nvidia_gpu;

using PropertiesParams = ov::AnyMap;

class ExecNetworkTest : public testing::Test,
                        public testing::WithParamInterface<PropertiesParams> {
    void SetUp() override {
        properties = this->GetParam();
        function_ = CreateMatMulTestNetwork();
        super_function_ = CreateSuperOperationTestNetwork();
    }

    void TearDown() override {}

public:
    static std::string getTestCaseName(testing::TestParamInfo<PropertiesParams> obj) {
        std::string target_device;
        PropertiesParams properties = obj.param;
        std::replace(target_device.begin(), target_device.end(), ':', '.');
        std::ostringstream result;
        result << "properties";
        for (auto& item : properties) {
            result << "_" << item.first << "=" << item.second.as<std::string>();
        }
        return result.str();
    }
    auto GetExecSequence(const std::shared_ptr<CompiledModel>& compiled_model) {
        const auto& graph = *compiled_model->graph_;
        std::vector<OperationBase::Ptr> execSequence{};
        execSequence.insert(execSequence.end(), graph.exec_sequence_.begin(), graph.exec_sequence_.end());
        return execSequence;
    }
    const auto& GetMemoryManagerPool(const std::shared_ptr<CompiledModel>& compiled_model) {
        return compiled_model->memory_pool_;
    }

    std::shared_ptr<ov::Model> function_;
    std::shared_ptr<ov::Model> super_function_;
    PropertiesParams properties;
};

std::vector<PropertiesParams> default_properties = {
    {
        {ov::device::id.name(), "0"},
    },
};

using MatMulExecNetworkTest = ExecNetworkTest;
TEST_P(MatMulExecNetworkTest, BuildExecutableSequence_MatMul_Success) {
    auto plugin = std::make_shared<Plugin>();
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(
        plugin->compile_model(function_, properties));
    const auto& execSequence = GetExecSequence(cuda_compiled_model);
    ASSERT_EQ(execSequence.size(), 3);
    ASSERT_EQ(std::type_index(typeid(*execSequence[1].get())), std::type_index(typeid(MatMulOp)));
}

INSTANTIATE_TEST_SUITE_P(ExecNetworkTest,
                         MatMulExecNetworkTest,
                         ::testing::ValuesIn(default_properties),
                         ExecNetworkTest::getTestCaseName);

using ExecutableSequenceExecNetworkTest = ExecNetworkTest;
TEST_P(ExecutableSequenceExecNetworkTest, BuildExecutableSequence_SuperOperation_Failed) {
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->compile_model(super_function_, properties), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(ExecNetworkTest,
                         ExecutableSequenceExecNetworkTest,
                         ::testing::ValuesIn(default_properties),
                         ExecNetworkTest::getTestCaseName);

std::vector<PropertiesParams> num_streams_1_properties = {
    {
        {CONFIG_KEY(DEVICE_ID), "0"},
        {NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS), "1"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::num_streams.name(), "1"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::num_streams.name(), "1"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::LATENCY)},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::LATENCY)},
        {ov::num_streams.name(), "1"},
    },
};

using NumStreams1ExecNetworkTest = ExecNetworkTest;
TEST_P(NumStreams1ExecNetworkTest, LoadExecNetwork_OptimalNumberInferRequests_1_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 1;
    auto compiled_model = plugin->compile_model(function_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
}

INSTANTIATE_TEST_SUITE_P(ExecNetworkTest,
                         NumStreams1ExecNetworkTest,
                         ::testing::ValuesIn(num_streams_1_properties),
                         ExecNetworkTest::getTestCaseName);

std::vector<PropertiesParams> num_streams_8_properties = {
    {
        {CONFIG_KEY(DEVICE_ID), "0"},
        {NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS), "8"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::num_streams.name(), "8"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::num_streams.name(), "8"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::LATENCY)},
        {ov::num_streams.name(), "8"},
    },
};

using NumStreams8ExecNetworkTest = ExecNetworkTest;
TEST_P(NumStreams8ExecNetworkTest, LoadExecNetwork_OptimalNumberInferRequests_8_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 8;
    auto compiled_model = plugin->compile_model(function_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
}

INSTANTIATE_TEST_SUITE_P(ExecNetworkTest,
                         NumStreams8ExecNetworkTest,
                         ::testing::ValuesIn(num_streams_8_properties),
                         ExecNetworkTest::getTestCaseName);


std::vector<PropertiesParams> num_streams_auto_properties = {
    {
        {CONFIG_KEY(DEVICE_ID), "0"},
        {NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS), NVIDIA_CONFIG_VALUE(THROUGHPUT_AUTO)},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::num_streams.name(), ov::util::to_string(ov::streams::AUTO)},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::THROUGHPUT)},
    }
};

using NumStreamsAUTOExecNetworkTest = ExecNetworkTest;
TEST_P(NumStreamsAUTOExecNetworkTest, LoadExecNetwork_OptimalNumberInferRequests_Auto_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    auto compiled_model = plugin->compile_model(function_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_GT(memoryManagerPool->Size(), 1);
}
