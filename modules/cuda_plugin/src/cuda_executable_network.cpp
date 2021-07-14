// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>
#include <utility>
#include <fmt/format.h>
#include <ops/nop_op.hpp>

#include "transformations/serialize.hpp"

#include "cuda/cuda_config.hpp"
#include "cuda_plugin.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"
#include "transformer/cuda_graph_transformer.hpp"

#include "ops/parameter.hpp"
#include "ops/result.hpp"
#include "memory_manager/model/cuda_memory_model_builder.hpp"
#include "memory_manager/cuda_immutable_memory_block_builder.hpp"
#include "memory_manager/cuda_memory_manager.hpp"

namespace CUDAPlugin {

using Time = std::chrono::steady_clock;

ExecutableNetwork::ExecutableNetwork(const InferenceEngine::CNNNetwork& cnnNetwork,
                                     Configuration cfg,
                                     InferenceEngine::ITaskExecutor::Ptr waitExecutor,
                                     Plugin::Ptr plugin) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, nullptr), // Disable default threads creation
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
        InitExecutor(); // creates thread-based executor using for async requests
        BenchmarkOptimalNumberOfRequests();
    } catch (const InferenceEngine::details::InferenceEngineException&) {
        throw;
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << fmt::format("Standard exception from compilation library: {}", e.what());
    } catch (...) {
        THROW_IE_EXCEPTION << "Generic exception is thrown";
    }
}

ExecutableNetwork::ExecutableNetwork(std::istream& model,
                                     Configuration cfg,
                                     InferenceEngine::ITaskExecutor::Ptr waitExecutor,
                                     Plugin::Ptr plugin) :
    cfg_(std::move(cfg)),
    cuda_stream_executor_(std::move(waitExecutor)),
    plugin_(std::move(plugin)) {
    // Read XML content
    std::string xmlString;
    std::uint64_t dataSize = 0;
    model.read(reinterpret_cast<char *>(&dataSize), sizeof(dataSize));
    xmlString.resize(dataSize);
    model.read(const_cast<char *>(xmlString.c_str()), dataSize);

    // Read blob content
    InferenceEngine::Blob::Ptr dataBlob;
    model.read(reinterpret_cast<char *>(&dataSize), sizeof(dataSize));
    if (0 != dataSize) {
        dataBlob = InferenceEngine::make_shared_blob<std::uint8_t>(
            InferenceEngine::TensorDesc(InferenceEngine::Precision::U8,
                                        {static_cast<std::size_t>(dataSize)},
                                        InferenceEngine::Layout::C));
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
        auto transformed_function = transformer.transform(
            CUDA::Device{cfg_.deviceId}, original_function, ConfigMap{});
        CompileNetwork(transformed_function);
        InitExecutor(); // creates thread-based executor using for async requests
        BenchmarkOptimalNumberOfRequests();
    } catch (const InferenceEngine::details::InferenceEngineException&) {
        throw;
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << fmt::format("Standard exception from compilation library: {}", e.what());
    } catch (...) {
        THROW_IE_EXCEPTION << "Generic exception is thrown";
    }
}

void ExecutableNetwork::CompileNetwork(const std::shared_ptr<const ngraph::Function>& function) {
    CUDA::Device device{cfg_.deviceId};
    function_ = function;
    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    for (auto&& result : function_->get_results()) {
        output_index_.emplace(ResultOp::GetOutputTensorName(*result), function_->get_result_index(result));
    }
    for (auto&& parameter : function_->get_parameters()) {
        input_index_.emplace(ParameterOp::GetInputTensorName(*parameter), function_->get_parameter_index(parameter));
    }

    const auto& orderedNodes = function_->get_ordered_ops();

    static constexpr auto InitNeeded = IOperationExec::WorkbufferStatus::InitNeeded;

    OperationBuffersExtractor opBuffersExtractor { orderedNodes };
    std::vector<OperationBase::Ptr> init_sequence {};
    // Perform any other steps like allocation and filling backend specific memory handles and so on
    const std::string optimizeOptionString = cfg_.Get(CUDA_CONFIG_KEY(OPTIMIZE));
    const bool optimizeOption = optimizeOptionString == CUDA_CONFIG_VALUE(YES);
    const auto creationContext = CUDA::CreationContext{device, optimizeOption};
    for (unsigned node_idx = 0; node_idx < orderedNodes.size(); node_idx++) {
        auto& node = orderedNodes[node_idx];
        if (!OperationRegistry::getInstance().hasOperation(node)) {
            THROW_IE_EXCEPTION << fmt::format(
                "Node: name = {}, description = {}; Is not found in OperationRegistry",
                node->get_name(), node->description());
        }
        auto inIds = opBuffersExtractor.inputBufferIndices(*node);
        auto outIds = opBuffersExtractor.outputBufferIndices(*node);
        auto operation = OperationRegistry::getInstance().createOperation(creationContext, node, move(inIds), move(outIds));
        if (InitNeeded == operation->SetWorkbufferIds(opBuffersExtractor.processWorkbufferRequest(node_idx, operation->GetWorkBufferRequest()))) {
          init_sequence.push_back(operation);
        }
        if (!dynamic_cast<NopOp*>(operation.get())) {
          exec_sequence_.push_back(operation);
        }
    }

    memory_manager_pool_ = CreateMemoryManagerPool(opBuffersExtractor);
    InitSharedImmutableWorkbuffers(init_sequence);
}

void ExecutableNetwork::BenchmarkOptimalNumberOfRequests() {
  const std::string throughputStreams = cfg_.Get(CUDA_CONFIG_KEY(THROUGHPUT_STREAMS));
  if (throughputStreams != CUDA_CONFIG_VALUE(THROUGHPUT_AUTO)) { return; }

  struct BenchmarkResult {
    unsigned numberOfInferRequests;
    unsigned fps;

    bool operator<(const BenchmarkResult& other) const {
      return other.fps < this->fps;
    }
  };

  auto start = Time::now();

  InferenceEngine::InferRequest warmupInferRequest{
      CreateBenchmarkInferRequest()};
  warmupInferRequest.Infer();

  auto numMemManagers = memory_manager_pool_->Size();
  std::mutex mtx;
  std::condition_variable cond_var;

  constexpr auto kTimesBenchmarkRun = 3;
  std::vector<BenchmarkResult> benchmarks;
  benchmarks.reserve(numMemManagers);
  for (unsigned numInfers = 1; numInfers <= numMemManagers; ++numInfers) {
    std::array<unsigned, kTimesBenchmarkRun> allFps{};
    std::for_each(allFps.begin(), allFps.end(),
                  [this, &mtx, &cond_var, &numInfers](auto& fps) {
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
  const auto avgFps = std::accumulate(
          optimalBenchmarks.begin(), optimalBenchmarks.end(), 0,
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
    memory_manager_pool_->Resize(optimalBenchmarkResult.numberOfInferRequests);
    fmt::print("Resize MemoryManagerPool from {} to {}\n", numMemManagers,
               optimalBenchmarkResult.numberOfInferRequests);
  }
  auto duration = Time::now() - start;
  auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
  fmt::print("Time of benchmark for optimal number infer-requests = {} ms\n",
             durationMs.count());
}

unsigned int ExecutableNetwork::RunBenchmarkFor(
    const int numInfers,
    std::mutex& mtx,
    std::condition_variable& cond_var) {
  std::unique_lock<std::mutex> lock{mtx};
  const auto start = Time::now();
  std::atomic<uint32_t> callbackCalled{0};
  std::vector<InferenceEngine::InferRequest> inferRequests(numInfers);
  for (int k = 0; k < numInfers; ++k) {
    inferRequests[k] = InferenceEngine::InferRequest{CreateBenchmarkInferRequest()};
  }
  for (int k = 0; k < numInfers; ++k) {
    inferRequests[k].SetCompletionCallback(
        [&callbackCalled, &cond_var] {
          callbackCalled.fetch_add(1, std::memory_order_acq_rel);
          cond_var.notify_one();
        });
    inferRequests[k].StartAsync();
  }
  cond_var.wait(lock, [&callbackCalled, &numInfers] {
    return numInfers == callbackCalled.load(std::memory_order_acquire);
  });
  const auto duration = Time::now() - start;
  const auto fps = numInfers * (std::chrono::seconds(1)/duration);
  return fps;
}

std::vector<InferenceEngine::gpu::DevicePointer<void*>>
ExecutableNetwork::getSharedWorkbuffers(const IOperationExec& operation) {
    std::vector<InferenceEngine::gpu::DevicePointer<void*>> result {};
    const auto& ids = operation.GetWorkbufferIds();
    for(const auto immutable_index : ids.immutableIndices) {
        void* ptr = immutable_workbuffers_->deviceTensorPtr(immutable_index);
        IE_ASSERT(ptr != nullptr) << "Workbuffer not found. ID is " << immutable_index;
        result.emplace_back(ptr);
    }
    return result;
}

void ExecutableNetwork::InitSharedImmutableWorkbuffers(const std::vector<OperationBase::Ptr>& init_sequence) {
    for(auto op : init_sequence) {
      op->InitSharedImmutableWorkbuffers(getSharedWorkbuffers(*op));
    }
    immutable_workbuffers_.reset();
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
    _callbackExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({"CudaCallbackExecutor"});
}

std::size_t ExecutableNetwork::GetOptimalNumberOfStreams(const std::size_t constBlobSize, const std::size_t memoryBlobSize) const {
    static constexpr std::size_t reasonable_limit_of_streams = 10;
    if (memoryBlobSize == 0) {
        THROW_IE_EXCEPTION << "Model is not loaded properly. Size of tensors for model is 0 !!";
    }
    CUDA::Device device{cfg_.deviceId};
    device.setCurrent();
    std::size_t free;
    [[maybe_unused]] std::size_t total;
    throwIfError(cudaMemGetInfo(&free, &total));
    const std::size_t maxStreamsSupported = maxConcurrentStreams(device);
    const auto availableInferRequests = (free - constBlobSize) / memoryBlobSize;
    if (0 == availableInferRequests) {
        THROW_IE_EXCEPTION << "Not enough memory even for single InferRequest !!";
    }

    const std::string throughputStreams = cfg_.Get(CUDA_CONFIG_KEY(THROUGHPUT_STREAMS));
    if (throughputStreams == CUDA_CONFIG_VALUE(THROUGHPUT_AUTO)) {
        return std::min({maxStreamsSupported, availableInferRequests, reasonable_limit_of_streams});
    } else {
        const std::size_t numStreams = std::stoi(throughputStreams);
        return std::min({ maxStreamsSupported, numStreams, availableInferRequests });
    }
}

std::shared_ptr<MemoryManagerPool>
ExecutableNetwork::CreateMemoryManagerPool(const OperationBuffersExtractor& opBuffersExtractor) {
    ImmutableMemoryBlockBuilder constants_block_builder;
    MemoryModelBuilder mutable_model_builder;
    ImmutableMemoryModelBuilder immutable_workbuffer_model_builder;

    // Process nGraph and add allocations
    for (auto index : opBuffersExtractor.immutableBuffersIndices()) {
        auto span = opBuffersExtractor.immutableBuffer(index);
        constants_block_builder.addAllocation(index, span.data(), span.size());
    }
    for (auto index : opBuffersExtractor.mutableBuffersIndices())
        mutable_model_builder.addAllocation(index,
                opBuffersExtractor.mutableBufferLifespanStart(index),
                opBuffersExtractor.mutableBufferLifespanEnd(index),
                opBuffersExtractor.mutableBufferSize(index));
    const auto& immutable_workbufer_sizes = opBuffersExtractor.immutableWorkbufferSizes();
    for (const auto& index : immutable_workbufer_sizes) {
        immutable_workbuffer_model_builder.addAllocation(index.first, index.second);
    }

    // Build memory model for mutable memory block
    auto memory_model = mutable_model_builder.build();

    auto immutable_workbuffer_model = immutable_workbuffer_model_builder.build();
    const auto constBlobSize = constants_block_builder.deviceMemoryBlockSize();
    const auto memoryBlobSize = memory_model->deviceMemoryBlockSize();
    const auto immutableWorkBuffersSize = immutable_workbuffer_model->deviceMemoryBlockSize();

    const auto numStreams = GetOptimalNumberOfStreams(constBlobSize + immutableWorkBuffersSize, memoryBlobSize);

    // Build shared constants memory block
    auto shared_constants_blob = constants_block_builder.build();

    immutable_workbuffers_ = std::make_shared<DeviceMemBlock>(immutable_workbuffer_model);
    // Later on, for each infer request
    return std::make_shared<MemoryManagerPool>(numStreams, shared_constants_blob, memory_model, immutable_workbuffers_);
}

int ExecutableNetwork::GetCudaDeviceId() const noexcept {
    const std::string deviceId = cfg_.Get(CONFIG_KEY(DEVICE_ID));
    return std::stoi(deviceId);
}

InferenceEngine::InferRequestInternal::Ptr
ExecutableNetwork::CreateBenchmarkInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
  return std::make_shared<CudaInferRequest>(
      networkInputs,
      networkOutputs,
      std::shared_ptr<ExecutableNetwork>(this, [](ExecutableNetwork*) {
      }));
}

InferenceEngine::IInferRequest::Ptr
ExecutableNetwork::CreateBenchmarkInferRequest() {
  InferenceEngine::IInferRequest::Ptr asyncRequest;
  auto internalRequest =
      CreateBenchmarkInferRequestImpl(_networkInputs, _networkOutputs);
  auto asyncThreadSafeImpl =
      std::make_shared<CudaAsyncInferRequest>(std::static_pointer_cast<CudaInferRequest>(internalRequest),
                                              _taskExecutor, cuda_stream_executor_, _callbackExecutor);
  asyncRequest.reset(new InferenceEngine::InferRequestBase(asyncThreadSafeImpl),
                     [](InferenceEngine::IInferRequest *p) { p->Release(); });
  //asyncRequest.reset(new InferenceEngine::InferRequestBase(asyncThreadSafeImpl));
  asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
  return asyncRequest;
}

InferenceEngine::InferRequestInternal::Ptr
ExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    return std::make_shared<CudaInferRequest>(networkInputs,
                                              networkOutputs,
                                              std::static_pointer_cast<ExecutableNetwork>(shared_from_this()));
}

InferenceEngine::IInferRequest::Ptr
ExecutableNetwork::CreateInferRequest() {
    InferenceEngine::IInferRequest::Ptr asyncRequest;
    auto internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    auto asyncThreadSafeImpl =
        std::make_shared<CudaAsyncInferRequest>(std::static_pointer_cast<CudaInferRequest>(internalRequest),
                                                _taskExecutor, cuda_stream_executor_, _callbackExecutor);
    asyncRequest.reset(new InferenceEngine::InferRequestBase(asyncThreadSafeImpl),
                       [](InferenceEngine::IInferRequest *p) { p->Release(); });
    //asyncRequest.reset(new InferenceEngine::InferRequestBase(asyncThreadSafeImpl));
    asyncThreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    return asyncRequest;
}

InferenceEngine::Parameter
ExecutableNetwork::GetConfig(const std::string& name) const {
    return cfg_.Get(name);
}

InferenceEngine::Parameter
ExecutableNetwork::GetMetric(const std::string& name) const {
    // TODO: return more supported values for metrics
    if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {
            CONFIG_KEY(DEVICE_ID),
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
        const unsigned value = memory_manager_pool_->Size();
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        THROW_IE_EXCEPTION << fmt::format("Unsupported ExecutableNetwork metric: {}", name);
    }
}

InferenceEngine::CNNNetwork
ExecutableNetwork::GetExecGraphInfo() {
    return cnn_network_;
}

void ExecutableNetwork::ExportImpl(std::ostream& modelStream) {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, "ExecutableNetwork::ExportImpl");

    // Note: custom ngraph extensions are not supported
    std::map<std::string, ngraph::OpSet> custom_opsets;
    std::stringstream xmlFile, binFile;
    ngraph::pass::Serialize serializer(xmlFile, binFile,
                                       ngraph::pass::Serialize::Version::IR_V10, custom_opsets);
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

} // namespace CUDAPlugin
