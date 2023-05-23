// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
// ^^ must come before ie_plugin_config.hpp, which is indirectly included by
// cuda_executable_network.hpp

#include <fmt/format.h>

#include <memory_manager/cuda_memory_manager.hpp>
#include <ops/nop_op.hpp>
#include <ops/subgraph.hpp>
#include <threading/ie_executor_manager.hpp>
#include <utility>

#include "cuda_compiled_model.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"
#include "cuda_plugin.hpp"
#include "memory_manager/cuda_immutable_memory_block_builder.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/model/cuda_memory_model_builder.hpp"
#include "nvidia/nvidia_config.hpp"
#include "nvidia/properties.hpp"

#include "ops/parameter.hpp"
#include "ops/result.hpp"
#include "transformations/utils/utils.hpp"
#include "transformer/cuda_graph_transformer.hpp"

#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/pass/serialize.hpp"

namespace {
static constexpr const char* nv_task_executor_name = "NvidiaCallbackExecutor";
static constexpr const char* nv_callback_executor_name = "NvidiaCallbackExecutor";
}  // namespace

namespace ov {
namespace nvidia_gpu {

using Time = std::chrono::steady_clock;

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const Configuration& cfg,
                             const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             bool loaded_from_cache)
    : ov::ICompiledModel(model, plugin, nullptr, nullptr),
      config_(std::move(cfg)),
      cuda_stream_executor_(std::move(wait_executor)),
      loaded_from_cache_(loaded_from_cache) {
    try {
        compile_model(model);
        init_executor();  // creates thread-based executor using for async requests
        benchmark_optimal_number_of_requests();
    } catch (const ov::Exception&) {
        throw;
    } catch (const InferenceEngine::Exception&) {
        throw;
    } catch (const std::exception& e) {
        throwIEException(fmt::format("Standard exception from compilation library: {}", e.what()));
    } catch (...) {
        throwIEException("Generic exception is thrown");
    }
}

void CompiledModel::init_executor() {
    // Default multi-threaded configuration is balanced for throughtput and latency cases and takes into account
    // real hardware cores and NUMA nodes.
    config_.streams_executor_config_.set_property({ ov::num_streams(ov::streams::Num(memory_pool_->Size())) });
    auto streams_executor_config = ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(config_.streams_executor_config_);
    streams_executor_config._name = nv_task_executor_name;
    // As Inference Engine CPU Streams Executor creates some additional therads
    // it is better to avoid threads recreateion as some OSs memory allocator can not manage such usage cases
    // and memory consumption can be larger than it is expected.
    // So Inference Engone provides exec    utors cache.
    set_task_executor(get_plugin()->get_executor_manager()->get_idle_cpu_streams_executor(streams_executor_config));
    set_callback_executor(get_plugin()->get_executor_manager()->get_idle_cpu_streams_executor({nv_callback_executor_name}));
}

CompiledModel::~CompiledModel() {
    get_plugin()->get_executor_manager()->clear(nv_task_executor_name);
    get_plugin()->get_executor_manager()->clear(nv_callback_executor_name);
}

void CompiledModel::compile_model(const std::shared_ptr<const ov::Model>& model) {
    CUDA::Device device{config_.get_device_id()};
    GraphTransformer transformer;
    // Clone model
    model_ = ov::clone_model(*model);
    // Apply transformations pipeline
    transformer.transform(device, model_, config_);
    // Generate backend specific blob mappings. For example Inference Engine uses not ov::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    for (auto&& result : model_->get_results()) {
        // TODO: Try to figure out why sometimes result_index >= device_function->get_results().size()
        const auto& result_index = model_->get_result_index(result->input_value(0));
        for (const auto& outputName : ResultOp::GetOutputTensorName(*result)) {
            output_index_.emplace(outputName, result_index);
        }
    }
    for (auto&& parameter : model_->get_parameters()) {
        const auto& parameter_index = model_->get_parameter_index(parameter);
        input_index_.emplace(ParameterOp::GetInputTensorName(*parameter), parameter_index);
    }

    // Integrate performance counters to the compiled model
    for (const auto& op : model_->get_ops()) {
        auto& rt_info = op->get_rt_info();
        rt_info[ov::nvidia_gpu::PERF_COUNTER_NAME] = std::make_shared<ov::nvidia_gpu::PerfCounts>();
    }

    // Perform any other steps like allocation and filling backend specific memory handles and so on
    const bool opBenchOption = config_.get(ov::nvidia_gpu::operation_benchmark.name()).as<bool>();
    const auto creationContext = CreationContext{device, opBenchOption};

    graph_ = std::make_unique<ExecGraph>(creationContext, model_);

    memory_pool_ = create_memory_pool();
}

void CompiledModel::benchmark_optimal_number_of_requests() {
    const auto throughputStreams = config_.get(ov::num_streams.name()).as<std::string>();
    if (throughputStreams != NVIDIA_CONFIG_VALUE(THROUGHPUT_AUTO)) {
        return;
    }

    struct BenchmarkResult {
        unsigned numberOfInferRequests;
        unsigned fps;

        bool operator<(const BenchmarkResult& other) const { return other.fps < this->fps; }
    };

    auto start = Time::now();

    create_benchmark_infer_request()->infer();

    auto numMemManagers = memory_pool_->Size();
    std::mutex mtx;
    std::condition_variable cond_var;

    constexpr auto kTimesBenchmarkRun = 3;
    std::vector<BenchmarkResult> benchmarks;
    benchmarks.reserve(numMemManagers);
    for (unsigned numInfers = 1; numInfers <= numMemManagers; ++numInfers) {
        std::array<unsigned, kTimesBenchmarkRun> allFps{};
        std::for_each(allFps.begin(), allFps.end(), [this, &mtx, &cond_var, &numInfers](auto& fps) {
            fps = run_benchmark_for(numInfers, mtx, cond_var);
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

unsigned int CompiledModel::run_benchmark_for(const int numInfers,
                                                std::mutex& mtx,
                                                std::condition_variable& cond_var) {
    std::unique_lock<std::mutex> lock{mtx};
    uint32_t callbackCalled = 0;
    std::vector<std::shared_ptr<ov::IAsyncInferRequest>> infer_requests;
    infer_requests.reserve(numInfers);
    for (int k = 0; k < numInfers; ++k) {
        infer_requests.push_back(create_benchmark_infer_request());
        infer_requests.back()->set_callback([&callbackCalled, &cond_var, &mtx](const std::exception_ptr&) {
            {
                std::lock_guard<std::mutex> lock{mtx};
                ++callbackCalled;
            }
            cond_var.notify_one();
        });
    }
    const auto start = Time::now();
    for (auto& e : infer_requests) {
        e->start_async();
    }
    cond_var.wait(lock, [&callbackCalled, &numInfers] { return numInfers == callbackCalled; });
    const auto duration = Time::now() - start;
    const auto fps = numInfers * (std::chrono::seconds(1) / duration);
    return fps;
}

std::size_t CompiledModel::get_optimal_number_of_streams(std::size_t const_blob_size,
                                                         std::size_t memory_blob_size) const {
    if (memory_blob_size == 0) {
        return 0;
    }
    CUDA::Device device{config_.get_device_id()};
    device.setCurrent();
    std::size_t free;
    [[maybe_unused]] std::size_t total;
    throwIfError(cudaMemGetInfo(&free, &total));
    const std::size_t max_streams_supported = max_concurrent_streams(device);
    const auto available_infer_requests = (free - const_blob_size) / memory_blob_size;
    if (0 == available_infer_requests) {
        throwIEException("Not enough memory even for single InferRequest !!");
    }
    const std::size_t num_streams = config_.get_optimal_number_of_streams();
    return std::min({max_streams_supported, available_infer_requests, num_streams});
}

std::shared_ptr<MemoryPool> CompiledModel::create_memory_pool() {
    const auto& memory_manager = graph_->memoryManager();
    const auto const_blob_size = memory_manager.immutableTensors().memoryModel()->deviceMemoryBlockSize();
    const auto immutable_work_buffers_size = memory_manager.immutableWorkbuffers().memoryModel()->deviceMemoryBlockSize();
    const auto& memory_model = memory_manager.mutableTensorsMemoryModel();
    const auto memory_blob_size = memory_model->deviceMemoryBlockSize();
    const auto num_streams = get_optimal_number_of_streams(const_blob_size + immutable_work_buffers_size, memory_blob_size);
    return std::make_shared<MemoryPool>(num_streams, memory_model);
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_benchmark_infer_request() const {
    auto internal_request = create_sync_infer_request();
    return std::make_shared<CudaAsyncInferRequest>(std::static_pointer_cast<CudaInferRequest>(std::move(internal_request)),
                                                   get_task_executor(),
                                                   cuda_stream_executor_,
                                                   get_callback_executor());
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<CudaInferRequest>(
        std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<CudaAsyncInferRequest>(
        std::static_pointer_cast<CudaInferRequest>(internal_request),
        get_task_executor(),
        cuda_stream_executor_,
        get_callback_executor());

    return async_infer_request;
}

void CompiledModel::set_property(const ov::AnyMap& properties) {
    config_ = Configuration{properties, config_};
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (ov::supported_properties == name) {
        std::vector<ov::PropertyName> supported_properties;
        supported_properties.push_back(ov::PropertyName(ov::supported_properties.name(), PropertyMutability::RO));
        supported_properties.push_back(ov::PropertyName(ov::model_name.name(), PropertyMutability::RO));
        supported_properties.push_back(ov::PropertyName(ov::execution_devices.name(), PropertyMutability::RO));
        supported_properties.push_back(
            ov::PropertyName(ov::optimal_number_of_infer_requests.name(), PropertyMutability::RO));
        supported_properties.push_back(ov::PropertyName(ov::loaded_from_cache.name(), PropertyMutability::RO));
        auto config_properties = config_.get_rw_properties();
        supported_properties.insert(supported_properties.end(), config_properties.begin(), config_properties.end());
        return decltype(ov::supported_properties)::value_type{supported_properties};
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             std::vector<std::string>{METRIC_KEY(NETWORK_NAME),
                                                      METRIC_KEY(SUPPORTED_METRICS),
                                                      METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                      METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)});
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        std::vector<std::string> configKeys = {CONFIG_KEY(DEVICE_ID),
                                               CONFIG_KEY(PERF_COUNT),
                                               CONFIG_KEY(CPU_THROUGHPUT_STREAMS),
                                               NVIDIA_CONFIG_KEY(THROUGHPUT_STREAMS)};
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configKeys.emplace_back(configKey);
        }
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (ov::model_name == name) {
        auto model_name = model_->get_friendly_name();
        return decltype(ov::model_name)::value_type{model_name};
    } else if (ov::optimal_number_of_infer_requests == name) {
        const unsigned value = memory_pool_->Size();
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_plugin()->get_device_name() + "." + std::to_string(config_.get_device_id())};
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{loaded_from_cache_};
    } else {
        return config_.get(name);
    }
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    auto model = model_->clone();
    // Add execution information into the model
    size_t exec_order = 0;
    for (const auto& op : model->get_ordered_ops()) {
        auto& info = op->get_rt_info();
        const auto& it = info.find(ov::nvidia_gpu::PERF_COUNTER_NAME);
        OPENVINO_ASSERT(it != info.end(), "Operation ", op, " doesn't contain performance counter");
        auto perf_count = it->second.as<std::shared_ptr<ov::nvidia_gpu::PerfCounts>>();
        OPENVINO_ASSERT(perf_count, "Performance counter is empty");
        info[ov::exec_model_info::LAYER_TYPE] = op->get_type_info().name;
        info[ov::exec_model_info::EXECUTION_ORDER] = std::to_string(exec_order++);
        info[ov::exec_model_info::IMPL_TYPE] = perf_count->impl_type;
        auto perf_count_enabled = config_.get(ov::enable_profiling.name()).as<bool>();
        info[ov::exec_model_info::PERF_COUNTER] = perf_count_enabled && perf_count->average() != 0
                                                      ? std::to_string(perf_count->average())
                                                      : "not_executed";

        std::string original_names = ov::getFusedNames(op);
        if (original_names.empty()) {
            original_names = op->get_friendly_name();
        } else if (original_names.find(op->get_friendly_name()) == std::string::npos) {
            original_names = op->get_friendly_name() + "," + original_names;
        }
        info[ov::exec_model_info::ORIGINAL_NAMES] = original_names;
        info[ov::exec_model_info::RUNTIME_PRECISION] = perf_count->runtime_precision;

        std::stringstream precisions_ss;
        for (size_t i = 0 ; i < op->get_output_size(); i++) {
            if (i > 0) precisions_ss << ",";
            precisions_ss << op->get_output_element_type(i);
        }
        info[ov::exec_model_info::OUTPUT_PRECISIONS] = precisions_ss.str();
    }
    return model;
}

void CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "CompiledModel::export_model");

    std::stringstream xml_file, bin_file;
    auto& rt_info = model_->get_rt_info();
    int64_t version = 11;
    if (rt_info.count("version")) {
        version = rt_info.at("version").as<int64_t>();
    }

    ov::pass::Serialize serializer(xml_file, bin_file, static_cast<ov::pass::Serialize::Version>(version));
    serializer.run_on_model(ov::clone_model(*model_));

    auto weights = bin_file.str();
    auto model = xml_file.str();

    auto data_size = static_cast<std::uint64_t>(model.size());
    model_stream.write(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    model_stream.write(model.c_str(), data_size);

    data_size = static_cast<std::uint64_t>(weights.size());
    model_stream.write(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    model_stream.write(reinterpret_cast<char*>(&weights[0]), data_size);
}

}  // namespace nvidia_gpu
}  // namespace ov