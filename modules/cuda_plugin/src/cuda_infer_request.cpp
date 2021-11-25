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
#include <precision_utils.h>

#include <algorithm>
#include <blob_transform.hpp>
#include <description_buffer.hpp>
#include <ie_parallel.hpp>
#include <map>
#include <memory>
#include <string>
#include <threading/ie_executor_manager.hpp>
#include <utility>

#include "cuda/cuda_config.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_itt.hpp"
#include "cuda_plugin.hpp"

using namespace InferenceEngine;

namespace CUDAPlugin {
using namespace utils;

using Time = std::chrono::steady_clock;

CudaInferRequest::CudaInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                   const InferenceEngine::OutputsDataMap& networkOutputs,
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork,
                                   bool isBenchmarkMode)
    : IInferRequestInternal(networkInputs, networkOutputs),
      _executableNetwork(executableNetwork),
      cancellation_token_{[this] { memory_proxy_.reset(); }},
      profiler_{_executableNetwork->cfg_.perfCount, *_executableNetwork->graph_},
      is_benchmark_mode_{isBenchmarkMode} {
    // TODO: allocate infer request device and host buffers if needed, fill
    // actual list of profiling tasks

    std::string name = _executableNetwork->newRequestName();
    _profilingTask = {
        openvino::itt::handle(name + "_Preprocess"),
        openvino::itt::handle(name + "_Postprocess"),
        openvino::itt::handle(name + "_StartPipline"),
        openvino::itt::handle(name + "_WaitPipline"),
    };

    for (auto&& [inputName, userInputInfo] : _networkInputs) {
        const auto& inputDescr = userInputInfo->getTensorDesc();
        auto userInputBlob = allocateBlob(inputDescr.getDims(), inputDescr.getPrecision(), inputDescr.getLayout());
        _inputs[inputName] = userInputBlob;
        const auto networkPrecision = convertType(_executableNetwork->parameter(inputName).get_element_type());
        if (inputDescr.getPrecision() != networkPrecision) {
            _deviceInputs[inputName] =
                allocateBlob(inputDescr.getDims(), Precision::FP32, TensorDesc::getLayoutByDims(inputDescr.getDims()));
        } else {
            _deviceInputs[inputName] = userInputBlob;
        }
        const auto& deviceInputDescr = _deviceInputs[inputName]->getTensorDesc();
        if (deviceInputDescr.getPrecision() != networkPrecision) {
            network_input_blobs_[inputName] =
                allocateBlob(inputDescr.getDims(), networkPrecision, TensorDesc::getLayoutByDims(inputDescr.getDims()));
        }
    }
    for (auto&& [outputName, userOutputInfo] : _networkOutputs) {
        const auto& descr = userOutputInfo->getTensorDesc();
        auto userOutputBlob = allocateBlob(descr.getDims(), descr.getPrecision(), descr.getLayout());
        _outputs[outputName] = userOutputBlob;
        const auto networkPrecision = convertType(_executableNetwork->result(outputName).get_element_type());
        if (descr.getPrecision() != networkPrecision) {
            network_output_blobs_[outputName] =
                allocateBlob(descr.getDims(), networkPrecision, TensorDesc::getLayoutByDims(descr.getDims()));
        } else {
            network_output_blobs_[outputName] = userOutputBlob;
        }
    }
}

void CudaInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Profiler::Preprocess]);
    cancellation_token_.Check();
    profiler_.StartStage();
    IInferRequestInternal::execDataPreprocessing(_deviceInputs);
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in
    //       constructor. That is why we assign network_input_blobs_[inputName]
    //       after InferRequestInternal::execDataPreprocessing(...)
    for (auto&& netInput : _networkInputs) {
        const auto& inputName = netInput.first;
        const auto networkPrecision = convertType(_executableNetwork->parameter(inputName).get_element_type());
        if (_deviceInputs[inputName]->getTensorDesc().getPrecision() == networkPrecision) {
            network_input_blobs_[inputName] = _deviceInputs[inputName];
        }
    }
    // InferRequestInternal::execDataPreprocessing() doesn't support conversion from fp32 to fp16.
    // It converts floats to int8_t values instead.
    // To avoid such behavior, CudaInferRequest creates network blob in fp32 format.
    // Subsequent conversion is performed here.
    for (auto&& [inputName, networkInput] : network_input_blobs_) {
        convertPrecision(_deviceInputs.at(inputName), networkInput);
    }
    cancellation_token_.Check();
    profiler_.StopStage(Profiler::Preprocess);
}

void CudaInferRequest::startPipeline(const ThreadContext& threadContext) {
    try {
        OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Profiler::StartPipeline])
        profiler_.StartStage();
        memory_proxy_ = _executableNetwork->memory_pool_->WaitAndGet(cancellation_token_);
        auto& memory = memory_proxy_->Get();
        auto& graph = *_executableNetwork->graph_;
        InferenceRequestContext inferRequestContext{
            network_input_blobs_, network_output_blobs_, threadContext, cancellation_token_, profiler_, is_benchmark_mode_};
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
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Profiler::WaitPipeline])
    cancellation_token_.Check();
    profiler_.StartStage();
    // TODO: probably all time will be spent in synchonize, out of reach of ThrowIfCanceled
    threadContext.stream().synchronize();
    memory_proxy_.reset();
    profiler_.StopStage(Profiler::WaitPipeline);
}

void CudaInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Profiler::Postprocess]);
    cancellation_token_.Check();
    profiler_.StartStage();
    for (auto&& output : _outputs) {
        cancellation_token_.Check();
        auto outputBlob = output.second;
        auto networkOutput = network_output_blobs_[output.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            convertPrecision(networkOutput, outputBlob);
        }
    }
    profiler_.StopStage(Profiler::Postprocess);
    profiler_.ProcessEvents();
}

void CudaInferRequest::Cancel() {
    cancellation_token_.Cancel();
    _executableNetwork->memory_pool_->Interrupt();
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

InferenceEngine::Precision::ePrecision CudaInferRequest::convertType(ngraph::element::Type_t type) {
    using InferenceEngine::Precision;
    using ngraph::element::Type_t;
    switch (type) {
        case Type_t::f16:
            return Precision::FP16;
        case Type_t::f32:
            return Precision::FP32;
        case Type_t::u8:
            return Precision::U8;
        case Type_t::i16:
            return Precision::I16;
        case Type_t::i32:
            return Precision::I32;
        case Type_t::boolean:
            return Precision::BOOL;
        default:
            throwIEException(fmt::format("Cuda Plugin: Unsupported Input/Output type {}", type));
    }
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

}  // namespace CUDAPlugin
