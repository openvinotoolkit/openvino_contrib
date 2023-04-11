// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_infer_request.hpp"

#include <cuda_fp16.h>
#include <debug.h>
#include <fmt/format.h>
#include <ie_blob.h>
#include <ie_layouts.h>
#include <ie_memcpy.h>

#include <algorithm>
#include <description_buffer.hpp>
#include <gsl/span_ext>
#include <map>
#include <memory>
#include <string>
#include <threading/ie_executor_manager.hpp>
#include <utility>

#include "cuda_executable_network.hpp"
#include "cuda_itt.hpp"
#include "cuda_plugin.hpp"
#include "ie_ngraph_utils.hpp"
#include "ngraph/util.hpp"

using namespace InferenceEngine;

namespace ov {
namespace nvidia_gpu {
using namespace utils;

using Time = std::chrono::steady_clock;

namespace {

template <typename BlobData, typename GetNetworkPrecisionF>
void allocateBlobImpl(BlobMap& blobMap,
                      BlobMap& networkBlobMap,
                      const BlobData& blobData,
                      GetNetworkPrecisionF&& getNetworkPrecision,
                      const SizeVector& dims,
                      bool isInputBlob) {
    const auto& precision = blobData.second->getTensorDesc().getPrecision();
    auto layout = blobData.second->getTensorDesc().getLayout();
    const auto deviceLayout = TensorDesc::getLayoutByDims(dims);
    Blob::Ptr& blob = blobMap[blobData.first];
    if (!blob) {
        blob = make_blob_with_precision({precision, dims, layout});
        blob->allocate();
    } else {
        blob->setShape(dims);
    }

    auto networkPrecision = InferenceEngine::details::convertPrecision(getNetworkPrecision(blobData.first));
    Blob::Ptr networkBlob;
    if (precision == networkPrecision && layout == deviceLayout) {
        networkBlob = blob;
    } else {
        if (isInputBlob) {
            networkBlob = make_blob_with_precision({InferenceEngine::Precision::FP32, dims, deviceLayout});
            networkBlob->allocate();
        } else {
            networkBlob = make_blob_with_precision({networkPrecision, dims, deviceLayout});
            networkBlob->allocate();
        }
    }
    networkBlobMap[blobData.first] = networkBlob;
}

template <typename BlobDataMap, typename GetNetworkPrecisionF>
void allocateBlobsImpl(const BlobDataMap& userDataMap,
                       BlobMap& userBlobMap,
                       BlobMap& deviceBlobMap,
                       GetNetworkPrecisionF&& getNetworkPrecision,
                       bool isInputBlob) {
    for (const auto& userData : userDataMap) {
        auto tensorDesc = userData.second->getTensorDesc();
        allocateBlobImpl(userBlobMap, deviceBlobMap, userData, getNetworkPrecision, tensorDesc.getDims(), isInputBlob);
    }
}

}  // namespace

CudaInferRequest::CudaInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                   const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork,
                                   bool isBenchmarkMode)
    : IInferRequestInternal(inputs, outputs),
      _executableNetwork(executableNetwork),
      cancellation_token_{[this] { memory_proxy_.reset(); }},
      profiler_{_executableNetwork->GetConfig(ov::enable_profiling.name()).as<bool>(), *_executableNetwork->graph_},
      is_benchmark_mode_{isBenchmarkMode} {
    this->setPointerToExecutableNetworkInternal(executableNetwork);
    createInferRequest();
}

CudaInferRequest::CudaInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                   const InferenceEngine::OutputsDataMap& networkOutputs,
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork,
                                   bool isBenchmarkMode)
    : IInferRequestInternal(networkInputs, networkOutputs),
      _executableNetwork(executableNetwork),
      cancellation_token_{[this] { memory_proxy_.reset(); }},
      profiler_{_executableNetwork->GetConfig(ov::enable_profiling.name()).as<bool>(), *_executableNetwork->graph_},
      is_benchmark_mode_{isBenchmarkMode} {
    this->setPointerToExecutableNetworkInternal(executableNetwork);
    createInferRequest();
}

void CudaInferRequest::allocateDeviceBuffers() {
    // Allocate plugin backend specific memory handles
    input_tensors_.resize(_networkInputs.size());
    output_tensors_.resize(_networkOutputs.size());
}

void CudaInferRequest::allocateBlobs() {
    auto&& parameters = _executableNetwork->function_->get_parameters();
    allocateBlobsImpl(
        _networkInputs,
        _inputs,
        _deviceInputs,
        [&](const std::string& blobName) {
            return parameters.at(_executableNetwork->input_index_.at(blobName))->get_element_type();
        },
        true);
    for (auto&& [inputName, userInputInfo] : _networkInputs) {
        const auto& inputDescr = userInputInfo->getTensorDesc();
        const auto networkPrecision =
            InferenceEngine::details::convertPrecision(_executableNetwork->parameter(inputName).get_element_type());
        const auto& deviceInputDescr = _deviceInputs[inputName]->getTensorDesc();
        if (deviceInputDescr.getPrecision() != networkPrecision) {
            Blob::Ptr networkBlob;
            networkBlob = make_blob_with_precision(
                {networkPrecision, inputDescr.getDims(), TensorDesc::getLayoutByDims(inputDescr.getDims())});
            networkBlob->allocate();
            network_input_blobs_[inputName] = networkBlob;
        }
    }
    auto&& results = _executableNetwork->function_->get_results();
    allocateBlobsImpl(
        _networkOutputs,
        _outputs,
        network_output_blobs_,
        [&](const std::string& blobName) {
            return results.at(_executableNetwork->output_index_.at(blobName))->get_element_type();
        },
        false);
}

void CudaInferRequest::createInferRequest() {
    // TODO: allocate infer request device and host buffers if needed, fill
    // actual list of profiling tasks

    std::string name = _executableNetwork->newRequestName();
    _profilingTask = {
        openvino::itt::handle(name + "_Preprocess"),
        openvino::itt::handle(name + "_Postprocess"),
        openvino::itt::handle(name + "_StartPipline"),
        openvino::itt::handle(name + "_WaitPipline"),
    };

    allocateDeviceBuffers();
    allocateBlobs();
}

void CudaInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, _profilingTask[Profiler::Preprocess]);
    cancellation_token_.Check();
    profiler_.StartStage();
    IInferRequestInternal::convertBatchedInputBlobs();
    IInferRequestInternal::execDataPreprocessing(_deviceInputs);
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can point to other memory region than it was allocated in
    //       constructor. That is why we assign network_input_blobs_[inputName]
    //       after InferRequestInternal::execDataPreprocessing(...)
    for (auto&& netInput : _networkInputs) {
        const auto& inputName = netInput.first;
        const auto networkPrecision =
            InferenceEngine::details::convertPrecision(_executableNetwork->parameter(inputName).get_element_type());
        if (_deviceInputs[inputName]->getTensorDesc().getPrecision() == networkPrecision) {
            network_input_blobs_[inputName] = _deviceInputs[inputName];
        }
    }
    for (auto&& networkInput : network_input_blobs_) {
        auto inputName = networkInput.first;
        auto index = _executableNetwork->input_index_.at(networkInput.first);
        const auto& parameter = _executableNetwork->function_->get_parameters().at(index);
        auto parameterShape = networkInput.second->getTensorDesc().getDims();
        auto srcShape = networkInput.second->getTensorDesc().getBlockingDesc().getBlockDims();
        const auto& parameterType = parameter->get_element_type();
        auto mem_blob = InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput.second);
        auto isNonRoiDesc = [](const BlockingDesc& desc) {
            size_t exp_stride = 1;
            const auto& blockDims = desc.getBlockDims();
            const auto& order = desc.getOrder();
            const auto& strides = desc.getStrides();
            const auto& offsetPaddingToData = desc.getOffsetPaddingToData();
            for (size_t i = 0; i < blockDims.size(); i++) {
                const size_t rev_idx = blockDims.size() - i - 1;
                OPENVINO_ASSERT(order.at(rev_idx) == rev_idx,
                                "ov::nvidia_gpu: unsupported tensors with mixed axes order: ",
                                ngraph::vector_to_string(order));
                if (strides.at(rev_idx) != exp_stride || offsetPaddingToData.at(rev_idx) != 0) {
                    return false;
                }
                exp_stride *= blockDims.at(rev_idx);
            }
            return true;
        };
        convertPrecision(_deviceInputs.at(inputName), networkInput.second);
        if (isNonRoiDesc(networkInput.second->getTensorDesc().getBlockingDesc())) {
            // No ROI extraction is needed
            input_tensors_.at(index) =
                std::make_shared<ngraph::HostTensor>(parameterType, parameterShape, mem_blob->rmap().as<void*>());
        } else {
            OPENVINO_ASSERT(parameterType.bitwidth() % 8 == 0,
                            "ov::nvidia_gpu: Unsupported ROI tensor with element type having ",
                            std::to_string(parameterType.bitwidth()),
                            " bits size");
            // Perform manual extraction of ROI tensor
            // Basic implementation doesn't take axis order into account `desc.getBlockingDesc().getOrder()`
            // Performance of manual extraction is not optimal, but it is ok for template implementation
            input_tensors_.at(index) = std::make_shared<ngraph::HostTensor>(parameterType, parameterShape);
            auto desc = mem_blob->getTensorDesc();
            auto* src_data = mem_blob->rmap().as<uint8_t*>();
            auto dst_tensor = std::dynamic_pointer_cast<ngraph::runtime::HostTensor>(input_tensors_.at(index));
            OPENVINO_ASSERT(dst_tensor, "nvidia_gpu error: Can't cast created tensor to HostTensor");
            auto* dst_data = dst_tensor->get_data_ptr<uint8_t>();
            std::vector<size_t> indexes(parameterShape.size());
            for (size_t dst_idx = 0; dst_idx < ov::shape_size(parameterShape); dst_idx++) {
                size_t val = dst_idx;
                size_t src_idx = 0;
                for (size_t j1 = 0; j1 < indexes.size(); j1++) {
                    size_t j = indexes.size() - j1 - 1;
                    indexes.at(j) = val % parameterShape.at(j) + desc.getBlockingDesc().getOffsetPaddingToData().at(j);
                    val /= parameterShape.at(j);
                    src_idx += indexes.at(j) * desc.getBlockingDesc().getStrides().at(j);
                }
                std::copy(src_data + src_idx * parameterType.size(),
                          src_data + (src_idx + 1) * parameterType.size(),
                          dst_data + dst_idx * parameterType.size());
            }
        }
    }
    for (auto&& output : _outputs) {
        auto outputBlob = output.second;
        auto networkOutput = network_output_blobs_.at(output.first);
        auto index = _executableNetwork->output_index_.at(output.first);
        if (outputBlob->getTensorDesc().getPrecision() == networkOutput->getTensorDesc().getPrecision()) {
            networkOutput = outputBlob;
        }
        const auto& result = _executableNetwork->function_->get_results()[index];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            output_tensors_.at(index) = std::make_shared<ngraph::HostTensor>();
            continue;
        }
        const auto& resultShape = result->get_shape();
        const auto& resultType = result->get_element_type();
        output_tensors_.at(index) = std::make_shared<ngraph::HostTensor>(
            resultType,
            resultShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkOutput)->wmap().as<void*>());
    }

    cancellation_token_.Check();
    profiler_.StopStage(Profiler::Preprocess);
}

void CudaInferRequest::startPipeline(const ThreadContext& threadContext) {
    try {
        OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, _profilingTask[Profiler::StartPipeline])
        profiler_.StartStage();
        memory_proxy_ = _executableNetwork->memory_pool_->WaitAndGet(cancellation_token_);
        auto& memory = memory_proxy_->Get();
        auto& graph = *_executableNetwork->graph_;
        InferenceRequestContext inferRequestContext{input_tensors_,
                                                    _executableNetwork->input_index_,
                                                    output_tensors_,
                                                    _executableNetwork->output_index_,
                                                    threadContext,
                                                    cancellation_token_,
                                                    profiler_,
                                                    is_benchmark_mode_};
        graph.Run(inferRequestContext, memory);
        profiler_.StopStage(Profiler::StartPipeline);
    } catch (...) {
        // TODO:
        // Log error once logger is available
        memory_proxy_.reset();
        throw;
    }
}

void CudaInferRequest::waitPipeline(const ThreadContext& threadContext) {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, _profilingTask[Profiler::WaitPipeline])
    cancellation_token_.Check();
    profiler_.StartStage();
    // TODO: probably all time will be spent in synchonize, out of reach of ThrowIfCanceled
    threadContext.stream().synchronize();
    memory_proxy_.reset();
    profiler_.StopStage(Profiler::WaitPipeline);
}

void CudaInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, _profilingTask[Profiler::Postprocess]);
    cancellation_token_.Check();
    profiler_.StartStage();
    for (auto&& output : _outputs) {
        auto index = _executableNetwork->output_index_[output.first];
        const auto& result = _executableNetwork->function_->get_results()[index];
        if (result->get_output_partial_shape(0).is_dynamic()) {
            // Touch blob to allocate it
            GetBlob(output.first);
        }
        auto outputBlob = _outputs.at(output.first);
        auto networkOutput = network_output_blobs_[output.first];
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            convertPrecision(networkOutput, outputBlob);
        } else if (result->get_output_partial_shape(0).is_dynamic()) {
            auto tensor = output_tensors_[_executableNetwork->output_index_.at(output.first)];
            tensor->read(InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob)->wmap().as<char*>(),
                         tensor->get_size_in_bytes());
        }
    }
    profiler_.StopStage(Profiler::Postprocess);
    profiler_.ProcessEvents();
}

void CudaInferRequest::Cancel() {
    cancellation_token_.Cancel();
    _executableNetwork->memory_pool_->Interrupt();
}

InferenceEngine::Blob::Ptr CudaInferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "GetBlob");
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    Blob::Ptr data;
    const SizeVector oneVector = {1};
    if (findInputAndOutputBlobByName(name, foundInput, foundOutput)) {
        // ROI blob is returned only if it was set previously. Otherwise default blob is returned.
        auto it = _preProcData.find(name);
        if (it != _preProcData.end()) {
            data = it->second->getRoiBlob();
        } else {
            data = _inputs[name];
            SizeVector dims;
            if (!data) {
                auto&& parameters = _executableNetwork->function_->get_parameters();
                const auto& pshape = parameters.at(_executableNetwork->input_index_.at(name))->get_partial_shape();
                dims = pshape.is_dynamic() ? SizeVector({0}) : pshape.get_shape();
                allocateBlobImpl(
                    _inputs,
                    _deviceInputs,
                    *_networkInputs.find(name),
                    [&](const std::string& blobName) {
                        return parameters.at(_executableNetwork->input_index_.at(blobName))->get_element_type();
                    },
                    dims,
                    true);
                const auto& userInputInfo = _networkInputs[name];
                const auto& inputDescr = userInputInfo->getTensorDesc();
                const auto networkPrecision =
                    InferenceEngine::details::convertPrecision(_executableNetwork->parameter(name).get_element_type());
                const auto& deviceInputDescr = _deviceInputs[name]->getTensorDesc();
                if (deviceInputDescr.getPrecision() != networkPrecision) {
                    Blob::Ptr networkBlob;
                    networkBlob = make_blob_with_precision(
                        {networkPrecision, inputDescr.getDims(), TensorDesc::getLayoutByDims(inputDescr.getDims())});
                    networkBlob->allocate();
                    network_input_blobs_[name] = networkBlob;
                }
                data = _inputs[name];
            } else {
                dims = data->getTensorDesc().getDims();
            }
            checkBlob(data, name, true, foundInput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
            auto& devBlob = _deviceInputs[name];
            if (preProcessingRequired(foundInput, data, devBlob)) {
                // if no devBlob, performs inplace
                addInputPreProcessingFor(name, data, devBlob ? devBlob : _inputs[name]);
            }
        }
    } else {
        data = _outputs[name];
        SizeVector dims;
        auto has_zeros = [](const SizeVector& vec) {
            return std::any_of(vec.cbegin(), vec.cend(), [](size_t e) { return e == 0; });
        };
        if (!has_zeros(foundOutput->getTensorDesc().getDims())) {
            dims = foundOutput->getTensorDesc().getDims();
        } else if (output_tensors_[_executableNetwork->output_index_.at(name)] &&
                   output_tensors_[_executableNetwork->output_index_.at(name)]->get_partial_shape().is_static()) {
            dims = output_tensors_[_executableNetwork->output_index_.at(name)]->get_shape();
        } else {
            auto rank = foundOutput->getTensorDesc().getDims().size();
            dims = SizeVector(rank == 0 ? 1 : rank, 0);
        }

        if (data->getTensorDesc().getDims() != dims) {
            auto&& results = _executableNetwork->function_->get_results();
            allocateBlobImpl(
                _outputs,
                network_output_blobs_,
                *_networkOutputs.find(name),
                [&](const std::string& blobName) {
                    return results.at(_executableNetwork->output_index_.at(blobName))->get_element_type();
                },
                dims,
                false);
            data = _outputs[name];
        }
        checkBlob(data, name, false, foundOutput->getTensorDesc().getLayout() != SCALAR ? dims : oneVector);
    }
    return data;
}

void CudaInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "SetBlob");
    if (name.empty()) {
        IE_THROW(NotFound) << "Failed to set blob with empty name";
    }
    if (!userBlob) IE_THROW(NotAllocated) << "Failed to set empty blob with name: \'" << name << "\'";
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;
    auto has_zeros = [](const SizeVector& vec) {
        return std::any_of(vec.cbegin(), vec.cend(), [](size_t e) { return e == 0; });
    };
    const bool isInput = findInputAndOutputBlobByName(name, foundInput, foundOutput);
    const bool compoundBlobPassed = userBlob->is<CompoundBlob>();
    const bool remoteBlobPassed = userBlob->is<RemoteBlob>();
    if (!compoundBlobPassed && !remoteBlobPassed && userBlob->buffer() == nullptr)
        IE_THROW(NotAllocated) << "Input data was not allocated. Input name: \'" << name << "\'";
    bool input_dynamic = foundInput && has_zeros(foundInput->getInputData()->getDims());
    bool output_dynamic = foundOutput && has_zeros(foundOutput->getDims());
    if (userBlob->size() == 0 && !(input_dynamic || output_dynamic)) {
        IE_THROW() << "Input data is empty. Input name: \'" << name << "\'";
    }

    size_t dataSize = userBlob->size();
    if (isInput) {
        // ilavreno: the condition below is obsolete, but we need an exact list of precisions
        // which are supports by G-API preprocessing
        if (foundInput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user input precision";
        }

        auto& devBlob = _deviceInputs[name];
        auto usrDims = userBlob->getTensorDesc().getDims();
        auto usrLayout = userBlob->getTensorDesc().getLayout();
        auto devDims = devBlob->getTensorDesc().getDims();
        auto devLayout = devBlob->getTensorDesc().getLayout();
        auto devPrecision = devBlob->getTensorDesc().getPrecision();
        if (input_dynamic && (devDims != usrDims || devLayout != usrLayout)) {
            devBlob = make_blob_with_precision({devPrecision, usrDims, TensorDesc::getLayoutByDims(usrDims)});
            devBlob->allocate();
            _deviceInputs[name] = devBlob;
        }
        const bool preProcRequired = preProcessingRequired(foundInput, userBlob, devBlob);
        if (compoundBlobPassed && !preProcRequired) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }

        if (preProcRequired) {
            addInputPreProcessingFor(name, userBlob, devBlob ? devBlob : _inputs[name]);
        } else {
            size_t inputSize = devBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                   ? InferenceEngine::details::product(devBlob->getTensorDesc().getDims())
                                   : 1;
            if (dataSize != inputSize) {
                IE_THROW() << "Input blob size is not equal network input size (" << dataSize << "!=" << inputSize
                           << ").";
            }
            _inputs[name] = userBlob;
            devBlob = userBlob;
        }
        _batched_inputs.erase(name);
    } else {
        if (compoundBlobPassed) {
            IE_THROW(NotImplemented) << "cannot set compound blob: supported only for input pre-processing";
        }
        auto& devBlob = network_output_blobs_[name];
        auto usrDims = userBlob->getTensorDesc().getDims();
        auto usrLayout = userBlob->getTensorDesc().getLayout();
        auto devDims = devBlob->getTensorDesc().getDims();
        auto devLayout = devBlob->getTensorDesc().getLayout();
        auto devPrecision = devBlob->getTensorDesc().getPrecision();
        if (output_dynamic && (devDims != usrDims || devLayout != usrLayout)) {
            devBlob = make_blob_with_precision({devPrecision, usrDims, TensorDesc::getLayoutByDims(usrDims)});
            devBlob->allocate();
            network_output_blobs_[name] = devBlob;
        }
        size_t outputSize = devBlob->getTensorDesc().getLayout() != InferenceEngine::Layout::SCALAR
                                ? details::product(devBlob->getTensorDesc().getDims())
                                : 1;
        if (dataSize != outputSize) {
            IE_THROW() << "Output blob size is not equal network output size (" << dataSize << "!=" << outputSize
                       << ").";
        }
        if (foundOutput->getPrecision() != userBlob->getTensorDesc().getPrecision()) {
            IE_THROW(ParameterMismatch)
                << "Failed to set Blob with precision not corresponding to user output precision";
        }
        _outputs[name] = userBlob;
    }
}

void CudaInferRequest::SetBlobsImpl(const std::string& name, const InferenceEngine::BatchedBlob::Ptr& batchedBlob) {
    _batched_inputs[name] = batchedBlob;
}

InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(const IOperationMeta& op, unsigned execution_index) {
    InferenceEngineProfileInfo result{};
    op.GetCategory().copy(result.exec_type, sizeof(result.exec_type) - 1);
    op.GetTypeName().copy(result.layer_type, sizeof(result.layer_type) - 1);
    result.execution_index = execution_index;
    return result;
}

InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(const std::string& layer,
                                                            const std::string_view& exec_type) {
    InferenceEngineProfileInfo result{};
    exec_type.copy(result.exec_type, sizeof(result.exec_type) - 1);
    layer.copy(result.layer_type, sizeof(result.layer_type) - 1);
    result.execution_index = 0;
    return result;
}

constexpr InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(long long realTime_uSec,
                                                                      long long cpuTime_uSec = 0) noexcept {
    return InferenceEngineProfileInfo{InferenceEngineProfileInfo::EXECUTED, realTime_uSec, cpuTime_uSec};
}

Profiler::PerformaceCounters CudaInferRequest::GetPerformanceCounts() const { return profiler_.GetPerformanceCounts(); }

std::shared_ptr<ExecutableNetwork> CudaInferRequest::GetExecNetwork() { return _executableNetwork; }

InferenceEngine::Blob::Ptr CudaInferRequest::allocateBlob(const std::vector<std::size_t>& shape,
                                                          InferenceEngine::Precision precision,
                                                          InferenceEngine::Layout layout) {
    Blob::Ptr blob;
    switch (precision) {
        case Precision::FP16:
            blob = InferenceEngine::make_shared_blob<std::uint16_t>({Precision::FP16, shape, layout});
            break;
        case Precision::FP32:
            blob = InferenceEngine::make_shared_blob<float>({Precision::FP32, shape, layout});
            break;
        case Precision::I16:
            blob = InferenceEngine::make_shared_blob<std::int16_t>({Precision::I16, shape, layout});
            break;
        case Precision::I32:
            blob = InferenceEngine::make_shared_blob<std::int32_t>({Precision::I32, shape, layout});
            break;
        case Precision::U8:
            blob = InferenceEngine::make_shared_blob<uint8_t>({Precision::U8, shape, layout});
            break;
        case Precision::BOOL:
            blob = InferenceEngine::make_shared_blob<std::uint8_t>({Precision::BOOL, shape, layout});
            break;
        default:
            throwIEException(fmt::format("Cuda Plugin: Unsupported Input/Output Precision {}", precision));
    }
    blob->allocate();
    return blob;
}

template <typename SrcT, typename DstT>
void CudaInferRequest::convertPrecision(const Blob::Ptr& src, const Blob::Ptr& dst) {
    std::copy_n(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                src->size(),
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>());
}

template <>
void CudaInferRequest::convertPrecision<__half, float>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const __half*>();
    std::transform(begin,
                   begin + src->size(),
                   InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<float*>(),
                   __half2float);
}

template <>
void CudaInferRequest::convertPrecision<float, __half>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const float*>();
    std::transform(begin,
                   begin + src->size(),
                   InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<__half*>(),
                   __float2half);
}

template <>
void CudaInferRequest::convertPrecision<std::uint8_t, __half>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const std::uint8_t*>();
    std::transform(begin,
                   begin + src->size(),
                   InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<__half*>(),
                   [](auto x) { return __float2half(static_cast<float>(x)); });
}

template <>
void CudaInferRequest::convertPrecision<__half, std::uint8_t>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const __half*>();
    std::transform(begin,
                   begin + src->size(),
                   InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<std::uint8_t*>(),
                   [](auto x) { return static_cast<std::uint8_t>(static_cast<float>(x)); });
}

template <>
void CudaInferRequest::convertPrecision<std::int16_t, __half>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const std::int16_t*>();
    std::transform(begin,
                   begin + src->size(),
                   InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<__half*>(),
                   [](auto x) { return __float2half(static_cast<float>(x)); });
}

template <>
void CudaInferRequest::convertPrecision<__half, std::int16_t>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const __half*>();
    std::transform(begin,
                   begin + src->size(),
                   InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<std::int16_t*>(),
                   [](auto x) { return static_cast<std::int16_t>(__half2float(x)); });
}

void CudaInferRequest::convertPrecision(const Blob::Ptr& src, const Blob::Ptr& dst) {
    if (src->getTensorDesc().getPrecision() == dst->getTensorDesc().getPrecision()) {
        return;
    }
    switch (src->getTensorDesc().getPrecision()) {
        case Precision::U8: {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP16:
                    convertPrecision<std::uint8_t, __half>(src, dst);
                    break;
                case Precision::FP32:
                    convertPrecision<std::uint8_t, float>(src, dst);
                    break;
                default: {
                    throwIEException(fmt::format("Unsupported precision conversion from {} to {}",
                                                 src->getTensorDesc().getPrecision(),
                                                 dst->getTensorDesc().getPrecision()));
                }
            }
        } break;
        case Precision::I16: {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP16:
                    convertPrecision<std::int16_t, __half>(src, dst);
                    break;
                case Precision::FP32:
                    convertPrecision<std::int16_t, float>(src, dst);
                    break;
                default: {
                    throwIEException(fmt::format("Unsupported precision conversion from {} to {}",
                                                 src->getTensorDesc().getPrecision(),
                                                 dst->getTensorDesc().getPrecision()));
                }
            }
        } break;
        case Precision::FP16:
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::U8:
                    convertPrecision<__half, std::uint8_t>(src, dst);
                    break;
                case Precision::FP32:
                    convertPrecision<__half, float>(src, dst);
                    break;
                default:
                    throwIEException(fmt::format("Unsupported precision conversion from {} to {}",
                                                 src->getTensorDesc().getPrecision(),
                                                 dst->getTensorDesc().getPrecision()));
            }
            break;
        case Precision::FP32: {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::U8:
                    convertPrecision<float, std::uint8_t>(src, dst);
                    break;
                case Precision::FP16:
                    convertPrecision<float, __half>(src, dst);
                    break;
                case Precision::I32:
                    convertPrecision<float, std::int32_t>(src, dst);
                    break;
                default: {
                    throwIEException(fmt::format("Unsupported precision conversion from {} to {}",
                                                 src->getTensorDesc().getPrecision(),
                                                 dst->getTensorDesc().getPrecision()));
                }
            }
        } break;
        case Precision::BOOL: {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP32:
                    convertPrecision<int8_t, float>(src, dst);
                    break;
                default: {
                    throwIEException(fmt::format("Unsupported precision conversion from {} to {}",
                                                 src->getTensorDesc().getPrecision(),
                                                 dst->getTensorDesc().getPrecision()));
                }
            }
        } break;
        default: {
            throwIEException(
                fmt::format("Unsupported precision conversion from {}", src->getTensorDesc().getPrecision()));
        }
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
