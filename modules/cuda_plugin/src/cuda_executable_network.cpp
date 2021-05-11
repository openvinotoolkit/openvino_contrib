// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_metric_helpers.hpp>
#include <ie_plugin_config.hpp>
#include <threading/ie_executor_manager.hpp>
#include <utility>
#include "transformations/serialize.hpp"

#include "cuda/cuda_config.hpp"
#include "cuda_plugin.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_itt.hpp"
#include "cuda_operation_registry.hpp"

#include "memory_manager/model/cuda_memory_model_builder.hpp"
#include "memory_manager/cuda_immutable_memory_block_builder.hpp"
#include "memory_manager/cuda_memory_manager.hpp"

// forward declaration
std::shared_ptr<ngraph::Function>
TransformNetwork(const std::shared_ptr<const ngraph::Function>& function);

namespace CUDAPlugin {

ExecutableNetwork::ExecutableNetwork(const InferenceEngine::CNNNetwork& cnnNetwork,
                                     Configuration cfg,
                                     InferenceEngine::IStreamsExecutor::Ptr waitExecutor,
                                     Plugin::Ptr plugin) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, nullptr), // Disable default threads creation
    cnn_network_(cnnNetwork),
    cfg_(std::move(cfg)),
    wait_executor_(std::move(waitExecutor)),
    plugin_(std::move(plugin)) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        CompileNetwork(cnn_network_.getFunction());
        InitExecutor(); // creates thread-based executor using for async requests
    } catch (const InferenceEngine::details::InferenceEngineException&) {
        throw;
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        THROW_IE_EXCEPTION << "Generic exception is thrown";
    }
}

ExecutableNetwork::ExecutableNetwork(std::istream& model,
                                     Configuration cfg,
                                     InferenceEngine::IStreamsExecutor::Ptr waitExecutor,
                                     Plugin::Ptr plugin) :
    cfg_(std::move(cfg)),
    wait_executor_(std::move(waitExecutor)),
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
        CompileNetwork(cnn_network_.getFunction());
        InitExecutor(); // creates thread-based executor using for async requests
    } catch (const InferenceEngine::details::InferenceEngineException&) {
        throw;
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << "Standard exception from compilation library: " << e.what();
    } catch (...) {
        THROW_IE_EXCEPTION << "Generic exception is thrown";
    }
}

void ExecutableNetwork::CompileNetwork(const std::shared_ptr<const ngraph::Function>& function) {
    // Apply plugins transformations
    function_ = TransformNetwork(function);

    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    for (auto&& result : function_->get_results()) {
        auto previousOutput = result->get_input_source_output(0);
        auto outputName = previousOutput.get_node()->get_friendly_name();
        if (previousOutput.get_node()->get_output_size() > 1) {
            outputName += '.' + std::to_string(previousOutput.get_index());
        }
        output_index_.emplace(outputName, function_->get_result_index(result));
    }
    for (auto&& parameter : function_->get_parameters()) {
        input_index_.emplace(parameter->get_friendly_name(), function_->get_parameter_index(parameter));
    }

    const auto& orderedNodes = function_->get_ordered_ops();
    tensor_collector_ = std::make_unique<TensorCollector>(orderedNodes);
    MemoryManagerComponentsSnippets();

    // Perform any other steps like allocation and filling backend specific memory handles and so on
    for (auto& node : orderedNodes) {
        if (!OperationRegistry::getInstance().hasOperation(node)) {
            THROW_IE_EXCEPTION << "Node: "
                               << "name = " << node->get_name() << ", "
                               << "description = " << node->description() << "; "
                               << "Is not found in OperationRegistry";
        }

        const auto&[inIds, outIds] = tensor_collector_->getTensorIds(node);
        auto operation = OperationRegistry::getInstance().createOperation(node, inIds, outIds);
        exec_sequence_.push_back(operation);
    }
}

void ExecutableNetwork::InitExecutor() {
    // Default multi-threaded configuration is balanced for throughtput and latency cases and takes into account
    // real hardware cores and NUMA nodes.
    auto streamsExecutorConfig =
        InferenceEngine::IStreamsExecutor::Config::MakeDefaultMultiThreaded(cfg_._streamsExecutorConfig);
    streamsExecutorConfig._name = "CudaCPUPreprocessExecutor";
    // As Inference Engine CPU Streams Executor creates some additional therads
    // it is better to avoid threads recreateion as some OSs memory allocator can not manage such usage cases
    // and memory consumption can be larger than it is expected.
    // So Inference Engone provides executors cache.
    _taskExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(streamsExecutorConfig);
    // NOTE: callback Executor is not configured. So callback will be called in the thread of the last stage of inference request pipeline
    // _callbackExecutor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({"CudaCallbackExecutor"});
}

void
ExecutableNetwork::MemoryManagerComponentsSnippets() {
    //
    // TODO: Remove. The following is a usage example.
    //

    ImmutableMemoryBlockBuilder constants_block_builder;
    MemoryModelBuilder mutable_model_builder;

    // Process nGraph and add allocations
    std::vector<uint8_t> rand_data(1000);
    constants_block_builder.addAllocation(0, &rand_data[0], rand_data.size());
    constants_block_builder.addAllocation(1, &rand_data[1], rand_data.size());
    mutable_model_builder.addAllocation(2, 0, 1, 988);
    mutable_model_builder.addAllocation(3, 0, 1, 988);

    // Build shared constants memory block
    shared_constants_blob_ = constants_block_builder.build();

    // Build memory model for mutable memory block
    memory_model_ = mutable_model_builder.build();

    // Later on, for each infer request
    const std::string numStreams = cfg_.Get(CUDA_CONFIG_KEY(THROUGHPUT_STREAMS));
    memory_manager_pool_ = std::make_shared<MemoryManagerPool>(std::stoi(numStreams), shared_constants_blob_, memory_model_);

    // For each operation
    // std::vector<MemoryModel::TensorID> inputIDs { 0, 1, 2 };
    // std::vector<MemoryModel::TensorID> outputIDs { 2, 3 };
    // auto inputs = memory_manager->inputTensorPointers(operation);
    // auto outputs = memory_manager->outputTensorPointers(operation);
}

std::shared_ptr<CudaStream>
ExecutableNetwork::GetCudaStream(const int streamId) {
    const std::string& deviceName = cfg_.Get(CONFIG_KEY(DEVICE_ID));
    auto& cudaStreams = plugin_->device_cuda_streams_[deviceName];
    auto foundStream = cudaStreams.find(streamId);
    if (cudaStreams.end() == foundStream) {
        THROW_IE_EXCEPTION << "Internal error: Could not find mapped CUDA Stream !!";
    }

    return foundStream->second;
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
                                                _taskExecutor, wait_executor_, _callbackExecutor);
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
        unsigned int value = cfg_._streamsExecutorConfig._streams;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, value);
    } else {
        THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
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
    serializer.run_on_function(function_);

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
