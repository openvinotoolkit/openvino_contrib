// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
// ^^ must come before ie_plugin_config.hpp, which is indirectly included by
// cuda_executable_network.hpp

#include <fmt/format.h>

#include <ie_icore.hpp>
#include <ie_plugin_config.hpp>
#include <memory_manager/cuda_memory_manager.hpp>
#include <ops/nop_op.hpp>
#include <ops/subgraph.hpp>
#include <threading/ie_executor_manager.hpp>
#include <utility>

#include "cuda/cuda_config.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "memory_manager/cuda_immutable_memory_block_builder.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/model/cuda_memory_model_builder.hpp"
#include "ops/parameter.hpp"
#include "ops/result.hpp"
#include "transformations/serialize.hpp"
#include "transformer/cuda_graph_transformer.hpp"

namespace CUDAPlugin {

using Time = std::chrono::steady_clock;

ExecutableNetwork::ExecutableNetwork(const InferenceEngine::CNNNetwork& cnnNetwork,
                                     Configuration cfg,
                                     InferenceEngine::ITaskExecutor::Ptr waitExecutor,
                                     Plugin::Ptr plugin)
    : InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, nullptr),  // Disable default threads creation
      cnn_network_(cnnNetwork),
      cfg_(std::move(cfg)),
      cuda_stream_executor_(std::move(waitExecutor)),
      plugin_(std::move(plugin)) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    setNetworkInputs(cnn_network_.getInputsInfo());
    setNetworkOutputs(cnn_network_.getOutputsInfo());
    try {
        CompileNetwork(cnn_network_.getFunction());
        InitExecutor();  // creates thread-based executor using for async requests
        BenchmarkOptimalNumberOfRequests();
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        throwIEException(fmt::format("Standard exception from compilation library: {}", e.what()));
    } catch (...) {
        throwIEException("Generic exception is thrown");
    }
}

ExecutableNetwork::ExecutableNetwork(std::istream& model,
                                     Configuration cfg,
                                     InferenceEngine::ITaskExecutor::Ptr waitExecutor,
                                     Plugin::Ptr plugin)
    : cfg_(std::move(cfg)), cuda_stream_executor_(std::move(waitExecutor)), plugin_(std::move(plugin)) {
    // Read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // Read blob content
    InferenceEngine::Blob::Ptr dataBlob;
    model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        dataBlob = InferenceEngine::make_shared_blob<std::uint8_t>(InferenceEngine::TensorDesc(
            InferenceEngine::Precision::U8, {static_cast<std::size_t>(dataSize)}, InferenceEngine::Layout::C));
        dataBlob->allocate();
        model.read(dataBlob->buffer(), dataSize);
    }

    // TODO: implement Import / Export of configuration options and merge with `cfg`
    // TODO: implement Import / Export of network precisions, layouts, preprocessing info

    cnn_network_ = plugin_->GetCore()->ReadNetwork(xmlString, std::move(dataBlob));

    setNetworkInputs(cnn_network_.getInputsInfo());
    setNetworkOutputs(cnn_network_.getOutputsInfo());
    SetPointerToPlugin(plugin_->shared_from_this());

    try {
        GraphTransformer transformer;
        auto original_function = cnn_network_.getFunction();
        auto transformed_function = transformer.transform(CUDA::Device{cfg_.deviceId}, original_function, cfg_);
        CompileNetwork(transformed_function);
        InitExecutor();  // creates thread-based executor using for async requests
        BenchmarkOptimalNumberOfRequests();
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        throwIEException(fmt::format("Standard exception from compilation library: {}", e.what()));
    } catch (...) {
        throwIEException("Generic exception is thrown");
    }
}

void ExecutableNetwork::CompileNetwork(const std::shared_ptr<const ngraph::Function>& function) {
    CUDA::Device device{cfg_.deviceId};
    function_ = function;
    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    for (auto&& result : function_->get_results()) {
        for (const auto& outputName : ResultOp::GetOutputTensorName(*result)) {
            const auto& result_index = function_->get_result_index(result);
            output_index_.emplace(outputName, result_index);
        }
    }
    for (auto&& parameter : function_->get_parameters()) {
        input_index_.emplace(ParameterOp::GetInputTensorName(*parameter), function_->get_parameter_index(parameter));
    }

    // Perform any other steps like allocation and filling backend specific memory handles and so on
    const std::string opBenchOptionString = cfg_.Get(CUDA_CONFIG_KEY(OPERATION_BENCHMARK));
    const bool opBenchOption = opBenchOptionString == CUDA_CONFIG_VALUE(YES);
    const auto creationContext = CreationContext{device, opBenchOption};

    graph_ = std::make_unique<CudaGraph>(creationContext, function_);

    memory_pool_ = CreateMemoryPool();
}

void ExecutableNetwork::BenchmarkOptimalNumberOfRequests() {
    const std::string throughputStreams = cfg_.Get(CUDA_CONFIG_KEY(THROUGHPUT_STREAMS));
    if (throughputStreams != CUDA_CONFIG_VALUE(THROUGHPUT_AUTO)) {
        return;
    }

    struct BenchmarkResult {
        unsigned numberOfInferRequests;
        unsigned fps;

        bool operator<(const BenchmarkResult& other) const { return other.fps < this->fps; }
    };

    auto start = Time::now();

    CreateBenchmarkInferRequest()->Infer();

    auto numMemManagers = memory_pool_->Size();
    std::mutex mtx;
    std::condition_variable cond_var;

    constexpr auto kTimesBenchmarkRun = 3;
    std::vector<BenchmarkResult> benchmarks;
    benchmarks.reserve(numMemManagers);
    for (unsigned numInfers = 1; numInfers <= numMemManagers; ++numInfers) {
        std::array<unsigned, kTimesBenchmarkRun> allFps{};
        std::for_each(allFps.begin(), allFps.end(), [this, &mtx, &cond_var, &numInfers](auto& fps) {
            fps = RunBenchmarkFor(numInfers, mtx, cond_var);
        });
        const unsigned fps = std::accumulate(allFps.begin(), allFps.end(), 0) / allFps.size();
        benchmarks.push_back({numInfers, fps});
    }
    std::sort(benchmarks.begin(), benchmarks.end(), std::less<>{});

    constexpr auto kNumberBestThroughputs = 3;
    std::vector<BenchmarkResult> optimalBenchmarks{};
    optimalBenchmarks.reserve(kNumberBestThroughputs);
    for (unsigned i = 0; i < kNumberBestThroughputs && i < benchmarks.size(); ++i) {
        optimalBenchmarks.push_back(benchmarks[i]);
    }

    constexpr auto kMaxFpsRelativeDiff = 0.01;
    const auto avgFps = std::accumulate(optimalBenchmarks.begin(),
                                        optimalBenchmarks.end(),
                                        0,
                                        [](const auto init, const auto& z) { return init + z.fps; }) /
                        optimalBenchmarks.size();
    const auto maxFpsDiff = kMaxFpsRelativeDiff * avgFps;

    auto optimalBenchmarkResult = optimalBenchmarks[0];
    for (auto& benchmark : optimalBenchmarks) {
        if (std::fabs(optimalBenchmarkResult.fps - benchmark.fps) < maxFpsDiff &&
            benchmark.numberOfInferRequests < optimalBenchmarkResult.numberOfInferRequests) {
            optimalBenchmarkResult = benchmark;
        }
    }
    fmt::print("Optimal number infer-requests = {}\n", optimalBenchmarkResult.numberOfInferRequests);
    if (optimalBenchmarkResult.numberOfInferRequests < numMemManagers) {
        memory_pool_->Resize(optimalBenchmarkResult.numberOfInferRequests);
        fmt::print(
            "Resize MemoryManagerPool from {} to {}\n", numMemManagers, optimalBenchmarkResult.numberOfInferRequests);
    }
    auto duration = Time::now() - start;
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    fmt::print("Time of benchmark for optimal number infer-requests = {} ms\n", durationMs.count());
}

unsigned int ExecutableNetwork::RunBenchmarkFor(const int numInfers,
                                                std::mutex& mtx,
                                                std::condition_variable& cond_var) {
    std::unique_lock<std::mutex> lock{mtx};
    uint32_t callbackCalled = 0;
    std::vector<InferenceEngine::IInferRequestInternal::Ptr> inferRequests;
    inferRequests.reserve(numInfers);
    for (int k = 0; k < numInfers; ++k) {
        inferRequests.push_back(CreateBenchmarkInferRequest());
        inferRequests.back()->SetCallback([&callbackCalled, &cond_var, &mtx](const std::exception_ptr&) {
            {
                std::lock_guard<std::mutex> lock{mtx};
                ++callbackCalled;
            }
            cond_var.notify_one();
        });
    }
    const auto start = Time::now();
    for (auto& e : inferRequests) {
        e->StartAsync();
    }
    cond_var.wait(lock, [&callbackCalled, &numInfers] { return numInfers == callbackCalled; });
    const auto duration = Time::now() - start;
    const auto fps = numInfers * (std::chrono::seconds(1) / duration);
    return fps;
}

void ExecutableNetwork::InitExecutor() {
    // Default multi-threaded configuration is balanced for throughtput and latency cases and takes into account
    // real hardware cores and NUMA nodes.
    auto streamsExecutorConfig =
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(cfg_.streams_executor_config_);
    streamsExecutorConfig._name = "CudaCPUPreprocessExecutor";
    // As Inference Engine CPU Streams Executor creates some additional therads
    // it is better to avoid threads recreateion as some OSs memory allocator can not manage such usage cases
    // and memory consumption can be larger than it is expected.
    // So Inference Engone provides executors cache.
    _taskExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    _callbackExecutor =
        InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({"CudaCallbackExecutor"});
}

std::size_t ExecutableNetwork::GetOptimalNumberOfStreams(const std::size_t constBlobSize,
                                                         const std::size_t memoryBlobSize) const {
    static constexpr std::size_t reasonable_limit_of_streams = 10;
    if (memoryBlobSize == 0) {
        throwIEException("Model is not loaded properly. Size of tensors for model is 0 !!");
    }
    CUDA::Device device{cfg_.deviceId};
    device.setCurrent();
    std::size_t free;
    [[maybe_unused]] std::size_t total;
    throwIfError(cudaMemGetInfo(&free, &total));
    const std::size_t maxStreamsSupported = maxConcurrentStreams(device);
    const auto availableInferRequests = (free - constBlobSize) / memoryBlobSize;
    if (0 == availableInferRequests) {
        throwIEException("Not enough memory even for single InferRequest !!");
    }

    const std::string throughputStreams = cfg_.Get(CUDA_CONFIG_KEY(THROUGHPUT_STREAMS));
    if (throughputStreams == CUDA_CONFIG_VALUE(THROUGHPUT_AUTO)) {
        return std::min({maxStreamsSupported, availableInferRequests, reasonable_limit_of_streams});
    } else {
        const std::size_t numStreams = std::stoi(throughputStreams);
        return std::min({maxStreamsSupported, numStreams, availableInferRequests});
    }
}

std::shared_ptr<MemoryPool> ExecutableNetwork::CreateMemoryPool() {
    const auto& memoryManager = graph_->memoryManager();
    const auto constBlobSize = memoryManager.immutableTensors().memoryModel()->deviceMemoryBlockSize();
    const auto immutableWorkBuffersSize = memoryManager.immutableWorkbuffers().memoryModel()->deviceMemoryBlockSize();
    const auto& memory_model = memoryManager.mutableTensorsMemoryModel();
    const auto memoryBlobSize = memory_model->deviceMemoryBlockSize();
    const auto numStreams = GetOptimalNumberOfStreams(constBlobSize + immutableWorkBuffersSize, memoryBlobSize);
    return std::make_shared<MemoryPool>(numStreams, memory_model);
}

int ExecutableNetwork::GetCudaDeviceId() const noexcept {
    const std::string deviceId = cfg_.Get(CONFIG_KEY(DEVICE_ID));
    return std::stoi(deviceId);
}

InferenceEngine::IInferRequestInternal::Ptr ExecutableNetwork::CreateBenchmarkInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<CudaInferRequest>(
        networkInputs, networkOutputs, std::shared_ptr<ExecutableNetwork>(this, [](ExecutableNetwork*) {}), true);
}

InferenceEngine::IInferRequestInternal::Ptr ExecutableNetwork::CreateBenchmarkInferRequest() {
    auto internalRequest = CreateBenchmarkInferRequestImpl(_networkInputs, _networkOutputs);
    return std::make_shared<CudaAsyncInferRequest>(std::static_pointer_cast<CudaInferRequest>(move(internalRequest)),
                                                   _taskExecutor,
                                                   cuda_stream_executor_,
                                                   _callbackExecutor);
}

InferenceEngine::IInferRequestInternal::Ptr ExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<CudaInferRequest>(
        networkInputs, networkOutputs, std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}

InferenceEngine::IInferRequestInternal::Ptr ExecutableNetwork::CreateInferRequest() {
    auto internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    return std::make_shared<CudaAsyncInferRequest>(std::static_pointer_cast<CudaInferRequest>(internalRequest),
                                                   _taskExecutor,
                                                   cuda_stream_executor_,
                                                   _callbackExecutor);
}

InferenceEngine::Parameter ExecutableNetwork::GetConfig(const std::string& name) const { return cfg_.Get(name); }

InferenceEngine::Parameter ExecutableNetwork::GetMetric(const std::string& name) const {
    // TODO: return more supported values for metrics
    if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             std::vector<std::string>{METRIC_KEY(NETWORK_NAME),
                                                      METRIC_KEY(SUPPORTED_METRICS),
                                                      METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                      METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {CONFIG_KEY(DEVICE_ID),
                                               CONFIG_KEY(PERF_COUNT),
                                               CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                               CUDA_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configKeys.emplace_back(configKey);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (EXEC_NETWORK_METRIC_KEY(NETWORK_NAME) == name) {
        auto networkName = function_->get_friendly_name();
        IE_SET_METRIC_RETURN(NETWORK_NAME, networkName);
    } else if (EXEC_NETWORK_METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS) == name) {
        const unsigned value = memory_pool_->Size();
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        throwIEException(fmt::format("Unsupported ExecutableNetwork metric: {}", name));
    }
}

InferenceEngine::CNNNetwork ExecutableNetwork::GetExecGraphInfo() { return cnn_network_; }

void ExecutableNetwork::Export(std::ostream& modelStream) {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, "ExecutableNetwork::Export");

    // Note: custom ngraph extensions are not supported
    std::map<std::string, ngraph::OpSet> custom_opsets;
    std::stringstream xmlFile, binFile;
    ngraph::pass::Serialize serializer(xmlFile, binFile, ngraph::pass::Serialize::Version::IR_V10, custom_opsets);
    serializer.run_on_function(ngraph::clone_function(*function_));

    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(m_model.size());
    modelStream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    modelStream.write(m_model.c_str(), dataSize);

    dataSize = static_cast<std::uint64_t>(m_constants.size());
    modelStream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    modelStream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);

    // TODO: implement network precision, layout, preprocessing info serialization
}

}  // namespace CUDAPlugin
