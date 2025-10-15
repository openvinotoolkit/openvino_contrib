// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ops/matmul.hpp>
#include <typeinfo>

#include "cuda_compiled_model.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "test_networks.hpp"

using namespace ov::nvidia_gpu;

using PropertiesParams = ov::AnyMap;

class CompileModelTest : public testing::Test, public testing::WithParamInterface<PropertiesParams> {
    void SetUp() override {
        properties = this->GetParam();
        model_ = create_matmul_test_model();
        super_model_ = create_super_operation_test_model();
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
        const auto& graph = compiled_model->get_topology_runner().GetSubGraph();
        std::vector<OperationBase::Ptr> execSequence{};
        auto graph_exec_sequence = graph.getExecSequence();
        execSequence.insert(execSequence.end(), graph_exec_sequence.begin(), graph_exec_sequence.end());
        return execSequence;
    }
    const auto& GetMemoryManagerPool(const std::shared_ptr<CompiledModel>& compiled_model) {
        return compiled_model->get_memory_pool();
    }

    std::shared_ptr<ov::Model> model_;
    std::shared_ptr<ov::Model> super_model_;
    PropertiesParams properties;
};

std::vector<PropertiesParams> default_properties = {
    {
        {ov::device::id.name(), "0"},
        {ov::hint::inference_precision.name(), "f16"},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::inference_precision.name(), "f32"},
    },
};

using MatMulCompileModelTest = CompileModelTest;
TEST_P(MatMulCompileModelTest, BuildExecutableSequence_MatMul_Success) {
    auto plugin = std::make_shared<Plugin>();
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(plugin->compile_model(model_, properties));
    const auto& execSequence = GetExecSequence(cuda_compiled_model);
    ASSERT_TRUE(execSequence.size() == 3 || execSequence.size() == 5);
    bool is_f32 = execSequence.size() == 3;
    auto matmul_index = is_f32 ? 1 : 2;
    ASSERT_EQ(std::type_index(typeid(*execSequence[matmul_index].get())), std::type_index(typeid(MatMulOp)));
}

INSTANTIATE_TEST_SUITE_P(CompileModelTest,
                         MatMulCompileModelTest,
                         ::testing::ValuesIn(default_properties),
                         CompileModelTest::getTestCaseName);

using ExecutableSequenceCompileModelTest = CompileModelTest;
TEST_P(ExecutableSequenceCompileModelTest, BuildExecutableSequence_SuperOperation_Failed) {
    auto plugin = std::make_shared<Plugin>();
    ASSERT_THROW(plugin->compile_model(super_model_, properties), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(CompileModelTest,
                         ExecutableSequenceCompileModelTest,
                         ::testing::ValuesIn(default_properties),
                         CompileModelTest::getTestCaseName);

std::vector<PropertiesParams> num_streams_1_properties = {
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

using NumStreams1CompileModelTest = CompileModelTest;
TEST_P(NumStreams1CompileModelTest, CompileModel_OptimalNumberInferRequests_1_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 1;
    auto compiled_model = plugin->compile_model(model_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
    ASSERT_EQ(cuda_compiled_model->get_property(ov::num_streams.name()), ov::streams::Num(total_streams));
    ASSERT_EQ(cuda_compiled_model->get_property(ov::optimal_number_of_infer_requests.name()), uint32_t(total_streams));
}

INSTANTIATE_TEST_SUITE_P(CompileModelTest,
                         NumStreams1CompileModelTest,
                         ::testing::ValuesIn(num_streams_1_properties),
                         CompileModelTest::getTestCaseName);

std::vector<PropertiesParams> num_streams_8_properties = {
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

using NumStreams8CompileModelTest = CompileModelTest;
TEST_P(NumStreams8CompileModelTest, CompileModel_OptimalNumberInferRequests_8_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 8;
    auto compiled_model = plugin->compile_model(model_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
    ASSERT_EQ(cuda_compiled_model->get_property(ov::num_streams.name()), ov::streams::Num(total_streams));
    ASSERT_EQ(cuda_compiled_model->get_property(ov::optimal_number_of_infer_requests.name()), uint32_t(total_streams));
}

INSTANTIATE_TEST_SUITE_P(CompileModelTest,
                         NumStreams8CompileModelTest,
                         ::testing::ValuesIn(num_streams_8_properties),
                         CompileModelTest::getTestCaseName);

std::vector<PropertiesParams> num_streams_8_properties_exclusive = {
    {
        {ov::device::id.name(), "0"},
        {ov::num_streams.name(), "8"},
        {ov::internal::exclusive_async_requests.name(), true},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::num_streams.name(), "8"},
        {ov::internal::exclusive_async_requests.name(), true},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::LATENCY)},
        {ov::num_streams.name(), "8"},
        {ov::internal::exclusive_async_requests.name(), true},
    },
};

using NumStreams8ExclusiveCompileModelTest = CompileModelTest;
TEST_P(NumStreams8ExclusiveCompileModelTest, CompileModel_OptimalNumberInferRequests_8_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    constexpr auto total_streams = 1;
    auto compiled_model = plugin->compile_model(model_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_EQ(memoryManagerPool->Size(), total_streams);
    ASSERT_EQ(cuda_compiled_model->get_property(ov::num_streams.name()), ov::streams::Num(total_streams));
    ASSERT_EQ(cuda_compiled_model->get_property(ov::optimal_number_of_infer_requests.name()), uint32_t(total_streams));
}

INSTANTIATE_TEST_SUITE_P(CompileModelTest,
                         NumStreams8ExclusiveCompileModelTest,
                         ::testing::ValuesIn(num_streams_8_properties_exclusive),
                         CompileModelTest::getTestCaseName);

std::vector<PropertiesParams> num_streams_auto_properties = {
    {
        {ov::device::id.name(), "0"},
        {ov::num_streams.name(), ov::util::to_string(ov::streams::AUTO)},
    },
    {
        {ov::device::id.name(), "0"},
        {ov::hint::performance_mode.name(), ov::util::to_string(ov::hint::PerformanceMode::THROUGHPUT)},
    }};

using NumStreamsAUTOCompileModelTest = CompileModelTest;
TEST_P(NumStreamsAUTOCompileModelTest, CompileModel_OptimalNumberInferRequests_Auto_Success) {
    using namespace std::chrono_literals;
    auto plugin = std::make_shared<Plugin>();
    auto compiled_model = plugin->compile_model(model_, properties);
    auto cuda_compiled_model = std::dynamic_pointer_cast<CompiledModel>(compiled_model);
    auto& memoryManagerPool = GetMemoryManagerPool(cuda_compiled_model);
    ASSERT_GT(memoryManagerPool->Size(), 1);
}

INSTANTIATE_TEST_SUITE_P(CompileModelTest,
                         NumStreamsAUTOCompileModelTest,
                         ::testing::ValuesIn(num_streams_auto_properties),
                         CompileModelTest::getTestCaseName);
