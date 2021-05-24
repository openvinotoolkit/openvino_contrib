// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <algorithm>
#include <memory>
#include <string>
#include <map>

#include <ie_blob.h>
#include <description_buffer.hpp>
#include <debug.h>
#include <ie_layouts.h>
#include <threading/ie_executor_manager.hpp>
#include <blob_transform.hpp>
#include <ie_parallel.hpp>
#include <ie_memcpy.h>
#include <precision_utils.h>

#include "cuda/cuda_config.hpp"
#include "cuda_infer_request.hpp"
#include "cuda_executable_network.hpp"
#include "cuda_plugin.hpp"
#include "cuda_itt.hpp"

#include <cuda_fp16.h>

using namespace InferenceEngine;

namespace CUDAPlugin {

using Time = std::chrono::steady_clock;

CudaInferRequest::CudaInferRequest(const InferenceEngine::InputsDataMap&                     networkInputs,
                                   const InferenceEngine::OutputsDataMap&                    networkOutputs,
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork),
    cancellation_token_{[this] {
        memory_manager_proxy_.reset();
    }} {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    std::string name = _executableNetwork->newRequestName();
    _profilingTask = {
      openvino::itt::handle(name + "_Preprocess"),
      openvino::itt::handle(name + "_Postprocess"),
      openvino::itt::handle(name + "_StartPipline"),
      openvino::itt::handle(name + "_WaitPipline"),
    };

    for (auto&& [inputName, userInputInfo] : _networkInputs) {
        const auto& descr = userInputInfo->getTensorDesc();
        auto userInputBlob = allocateBlob(descr.getDims(), descr.getPrecision(), descr.getLayout());
        _inputs[inputName] = userInputBlob;
        const auto networkPrecision = convertType(_executableNetwork->parameter(inputName).get_element_type());
        if (descr.getPrecision() != networkPrecision) {
            if (descr.getPrecision() == Precision::FP32 && networkPrecision == Precision::FP16) {
                // Let InferRequestInternal::execDataPreprocessing do preprocessing without type conversion.
                // Type conversion will be performed in CudaInferRequest::inferPreprocess().
                _deviceInputs[inputName] = userInputBlob;
                fp16NetworkInputBlobs_[inputName] = allocateBlob(descr.getDims(),
                        networkPrecision, TensorDesc::getLayoutByDims(descr.getDims()));
            } else {
            _deviceInputs[inputName] = allocateBlob(descr.getDims(),
                    networkPrecision, TensorDesc::getLayoutByDims(descr.getDims()));
            }
        } else {
            _deviceInputs[inputName] = userInputBlob;
        }
    }
    for (auto&& [outputName, userOutputInfo] : _networkOutputs) {
        const auto& descr = userOutputInfo->getTensorDesc();
        auto userOutputBlob = allocateBlob(descr.getDims(), descr.getPrecision(), descr.getLayout());
        _outputs[outputName] = userOutputBlob;
        const auto networkPrecision = convertType(_executableNetwork->result(outputName).get_element_type());
        if (descr.getPrecision() != networkPrecision)
            _networkOutputBlobs[outputName] = allocateBlob(descr.getDims(),
                    networkPrecision, TensorDesc::getLayoutByDims(descr.getDims()));
        else
            _networkOutputBlobs[outputName] = userOutputBlob;
    }
}

void CudaInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Preprocess]);
    cancellation_token_.Check();
    auto start = Time::now();
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    InferRequestInternal::execDataPreprocessing(_deviceInputs);
    // InferRequestInternal::execDataPreprocessing() doesn't support conversion from fp32 to fp16.
    // It converts floats to int8_t values instead.
    // To avoid such behavior, CudaInferRequest creates network blob in fp32 format.
    // Subsequent conversion is performed here.
    for (auto&& [inputName, fp16NetworkInput] : fp16NetworkInputBlobs_) {
        convertPrecision(_deviceInputs.at(inputName), fp16NetworkInput);
        _deviceInputs[inputName] = fp16NetworkInput;
    }
    cancellation_token_.Check();
    _durations[Preprocess] = Time::now() - start;
}

void CudaInferRequest::startPipeline(const CUDA::ThreadContext& threadContext) {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[StartPipeline])
    auto start = Time::now();
    memory_manager_proxy_ =
        _executableNetwork->memory_manager_pool_->WaitAndGet(cancellation_token_);
    auto& manager = memory_manager_proxy_->Get();
    InferenceRequestContext inferRequestContext{_deviceInputs, _networkOutputBlobs, threadContext};
    for (auto& op : _executableNetwork->exec_sequence_) {
      cancellation_token_.Check();
      auto inputTensors = manager.inputTensorPointers(*op);
      auto outputTensors = manager.outputTensorPointers(*op);
      op->Execute(inferRequestContext, inputTensors, outputTensors);
    }
    _durations[StartPipeline] = Time::now() - start;
}

void CudaInferRequest::waitPipeline(const CUDA::ThreadContext& threadContext) {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[WaitPipeline])
    cancellation_token_.Check();
    auto start = Time::now();
    // TODO: probably all time will be spent in synchonize, out of reach of ThrowIfCanceled
    threadContext.stream().synchronize();
    memory_manager_proxy_.reset();
    _durations[WaitPipeline] = Time::now() - start;
}

void CudaInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Postprocess]);
    cancellation_token_.Check();
    auto start = Time::now();
    for (auto&& output : _outputs) {
        cancellation_token_.Check();
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            convertPrecision(networkOutput, outputBlob);
        }
    }
    _durations[Postprocess] = Time::now() - start;
}

void CudaInferRequest::Cancel() {
    cancellation_token_.Cancel();
    _executableNetwork->memory_manager_pool_->Interrupt();
}

std::map<std::string, InferenceEngineProfileInfo> CudaInferRequest::GetPerformanceCounts() const {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    InferenceEngineProfileInfo info;
    info.execution_index = 0;
    info.status = InferenceEngineProfileInfo::EXECUTED;

    info.cpu_uSec = info.realTime_uSec = _durations[Preprocess].count();
    perfMap["1. input preprocessing"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["2. input transfer to a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[StartPipeline].count();
    perfMap["3. execution time"] = info;
    info.cpu_uSec = info.realTime_uSec = 0;
    perfMap["4. output transfer from a device"] = info;
    info.cpu_uSec = info.realTime_uSec = _durations[Postprocess].count();
    perfMap["5. output postprocessing"] = info;
    return perfMap;
}

std::shared_ptr<ExecutableNetwork>
CudaInferRequest::GetExecNetwork() {
    return _executableNetwork;
}

InferenceEngine::Blob::Ptr CudaInferRequest::allocateBlob(
        const std::vector<std::size_t>& shape, InferenceEngine::Precision precision,
        InferenceEngine::Layout layout) {
    Blob::Ptr blob;
    switch (precision) {
    case Precision::FP16:
        blob = InferenceEngine::make_shared_blob<std::uint16_t>({
                Precision::FP16, shape, layout });
        break;
    case Precision::FP32:
        blob = InferenceEngine::make_shared_blob<float>({
                Precision::FP32, shape, layout });
        break;
    case Precision::I16:
        blob = InferenceEngine::make_shared_blob<std::int16_t>({
                Precision::I16, shape, layout });
        break;
    case Precision::U8:
        blob = InferenceEngine::make_shared_blob<uint8_t>({
                Precision::U8, shape, layout });
        break;
    default:
        THROW_IE_EXCEPTION << "Cuda Plugin: Unsupported Input/Output Precision " << precision;
    }
    blob->allocate();
    return blob;
}

InferenceEngine::Precision::ePrecision CudaInferRequest::convertType(
        ngraph::element::Type_t type) {
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
    default:
        THROW_IE_EXCEPTION << "Cuda Plugin: Unsupported Input/Output type " << type;
    }
}

template<typename SrcT, typename DstT>
void CudaInferRequest::convertPrecision(const Blob::Ptr& src, const Blob::Ptr& dst) {
    std::copy_n(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                src->size(),
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>());
}

template<>
void CudaInferRequest::convertPrecision<__half, float>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const __half*>();
    std::transform(begin,
            begin + src->size(),
            InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<float*>(),
            __half2float);
}

template<>
void CudaInferRequest::convertPrecision<float, __half>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const float*>();
    std::transform(begin,
            begin + src->size(),
            InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<__half*>(),
            __float2half);
}

void CudaInferRequest::convertPrecision(const Blob::Ptr& src, const Blob::Ptr& dst) {
    if (src->getTensorDesc().getPrecision() == dst->getTensorDesc().getPrecision())
        return;
    switch (src->getTensorDesc().getPrecision()) {
        case Precision::U8 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP32 :
                    convertPrecision<std::uint8_t, float>(src, dst);
                    break;
                default : {
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        case Precision::FP16:
            switch (dst->getTensorDesc().getPrecision()) {
            case Precision::FP32:
                convertPrecision<__half, float>(src, dst);
                break;
            default:
                THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                    << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
            }
            break;
        case Precision::FP32 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::U8 :
                    convertPrecision<float, std::uint8_t>(src, dst);
                    break;
                case Precision::FP16:
                    convertPrecision<float, __half>(src, dst);
                    break;
                default : {
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        default : {
            THROW_IE_EXCEPTION << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision();
        }
    }
}

} // namespace CUDAPlugin
