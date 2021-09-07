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
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork)
    : IInferRequestInternal(networkInputs, networkOutputs),
      _executableNetwork(executableNetwork),
      cancellation_token_{[this] { memory_manager_proxy_.reset(); }} {
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
        auto userInputBlob = allocateBlob(inputDescr.getDims(), inputDescr.getPrecision(),
                         inputDescr.getLayout());
        _inputs[inputName] = userInputBlob;
        const auto networkPrecision = convertType(_executableNetwork->parameter(inputName).get_element_type());
        if (inputDescr.getPrecision() != networkPrecision) {
            _deviceInputs[inputName] = allocateBlob(
                inputDescr.getDims(),
                Precision::FP32,
                TensorDesc::getLayoutByDims(inputDescr.getDims()));
        } else {
            _deviceInputs[inputName] = userInputBlob;
        }
        const auto& deviceInputDescr = _deviceInputs[inputName]->getTensorDesc();
        if (deviceInputDescr.getPrecision() != networkPrecision) {
            network_input_blobs_[inputName] = allocateBlob(
                inputDescr.getDims(),
                networkPrecision,
                TensorDesc::getLayoutByDims(inputDescr.getDims()));
        }
    }
    for (auto&& [outputName, userOutputInfo] : _networkOutputs) {
        const auto& descr = userOutputInfo->getTensorDesc();
        auto userOutputBlob = allocateBlob(descr.getDims(), descr.getPrecision(), descr.getLayout());
        _outputs[outputName] = userOutputBlob;
        const auto networkPrecision = convertType(_executableNetwork->result(outputName).get_element_type());
        if (descr.getPrecision() != networkPrecision) {
            network_output_blobs_[outputName] = allocateBlob(
                descr.getDims(),
                networkPrecision,
                TensorDesc::getLayoutByDims(descr.getDims()));
        } else {
            network_output_blobs_[outputName] = userOutputBlob;
        }
    }
}

void CudaInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Preprocess]);
    cancellation_token_.Check();
    auto start = Time::now();
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
    _durations[Preprocess] = Time::now() - start;
}

void CudaInferRequest::startPipeline(const CUDA::ThreadContext& threadContext) {
    try {
        const bool perfCount = _executableNetwork->cfg_.perfCount;
        OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[StartPipeline])
        infer_count_++;
        auto start = Time::now();
        memory_manager_proxy_ =
                _executableNetwork->memory_manager_pool_->WaitAndGet(cancellation_token_);
        auto& manager = memory_manager_proxy_->Get();
        InferenceRequestContext inferRequestContext{network_input_blobs_, network_output_blobs_, threadContext};
        unsigned execution_index {};
        if (perfCount) exec_timing_.setStart(threadContext.stream());
        for (auto& op : _executableNetwork->exec_sequence_) {
            cancellation_token_.Check();
            auto inputTensors = manager.inputTensorPointers(*op);
            auto outputTensors = manager.outputTensorPointers(*op);
            if (perfCount) addStartEvent(threadContext.stream(), *op, ++execution_index);
            op->Execute(inferRequestContext, inputTensors, outputTensors, manager.workBuffers(*op));
            if (perfCount) addStopEvent(threadContext.stream(), *op);
        }
        if (perfCount) exec_timing_.setStop(threadContext.stream());
        _durations[StartPipeline] = Time::now() - start;
    } catch(...) {
        // TODO:
        // Log error once logger is available
        memory_manager_proxy_.reset();
        throw;
    }
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
        auto networkOutput = network_output_blobs_[output.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            convertPrecision(networkOutput, outputBlob);
        }
    }
    _durations[Postprocess] = Time::now() - start;
    processPerfEvents();
}

void CudaInferRequest::Cancel() {
    cancellation_token_.Cancel();
    _executableNetwork->memory_manager_pool_->Interrupt();
}
InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(const IOperationMeta& op, unsigned execution_index) {
  InferenceEngineProfileInfo result {};
  op.GetCategory().copy(result.exec_type, sizeof(result.exec_type)-1);
  op.GetTypeName().copy(result.layer_type, sizeof(result.layer_type)-1);
  result.execution_index = execution_index;
  return result;
}

InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(const std::string& layer, const std::string_view& exec_type) {
  InferenceEngineProfileInfo result {};
  exec_type.copy(result.exec_type, sizeof(result.exec_type)-1);
  layer.copy(result.layer_type, sizeof(result.layer_type)-1);
  result.execution_index = 0;
  return result;
}

constexpr InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(long long realTime_uSec, long long cpuTime_uSec = 0) noexcept {
  return InferenceEngineProfileInfo {
    InferenceEngineProfileInfo::EXECUTED,
    realTime_uSec,
    cpuTime_uSec
  };
}

void CudaInferRequest::addStartEvent(const CUDA::Stream& stream, const IOperationMeta& op, unsigned index) {
  const auto& name = op.GetName();
  const auto& type = op.GetTypeName();
  auto perf = perf_counters_.find(name);
  if (perf == perf_counters_.cend())
    perf_counters_.emplace(name, makeProfileInfo(op, index));
  perf = perf_counters_.find(type);
  if (perf == perf_counters_.cend()) {
    perf_counters_.emplace(type, makeProfileInfo(op.GetTypeName(), op.GetCategory()));
  } else {
    // Layers of the same type may have different exec_type, in sych case we clear exec_type
    if (perf->second.exec_type[0] && op.GetCategory().compare(perf->second.exec_type) != 0)
      perf->second.exec_type[0] = 0;
  }
  const auto timing = perf_timings_.find(name);
  if (timing == perf_timings_.cend()) {
    perf_timings_.emplace(name, PerformaceTiming{stream});
  } else {
    timing->second.setStart(stream);
  }
}

void CudaInferRequest::addStopEvent(const CUDA::Stream& stream, const IOperationMeta& op) {
  auto timing = perf_timings_.find(op.GetName());
  if (timing != perf_timings_.cend())
    timing->second.setStop(stream);
}

void CudaInferRequest::clearPerfEvents() {
  for (auto& t : perf_timings_) {
    t.second.clear();
  }
}

CudaInferRequest::PerformaceCounters CudaInferRequest::GetPerformanceCounts() const {
  return perf_counters_;
}

void CudaInferRequest::processPerfEvents() {
  if (infer_count_ == 0) return;
  constexpr float ms2us = 1000.0;
  std::map<std::string, float> layer_timing {};
  for (auto& timing : perf_timings_) {
    timing.second.measure();
    const auto perf = perf_counters_.find(timing.first);
    if (perf != perf_counters_.cend()) {
       perf->second.realTime_uSec = timing.second.duration() * ms2us / infer_count_;
       perf->second.status = InferenceEngineProfileInfo::EXECUTED;
       if (perf->second.layer_type[0]) {
         layer_timing[perf->second.layer_type] += timing.second.duration();
       }
    }
  }
  for (auto const& timing : layer_timing) {
    const auto summary = perf_counters_.find(timing.first);
    if (summary != perf_counters_.cend()) {
        summary->second.realTime_uSec = timing.second * ms2us / infer_count_;
        summary->second.status = InferenceEngineProfileInfo::EXECUTED;
    }
  }

  auto param_timing = layer_timing.find("Parameter");
  auto result_timing = layer_timing.find("Result");
  // Adding some overall performance counters
  perf_counters_["1. input preprocessing"] = makeProfileInfo(0, _durations[Preprocess].count());
  perf_counters_["2. input transfer to a device"] = makeProfileInfo(
      // Sum of all Parameters divided by count of infer requests
      param_timing == layer_timing.cend() ? 0 : param_timing->second * ms2us / infer_count_);
  perf_counters_["3. execution time"] = makeProfileInfo(
      exec_timing_.measure() * ms2us / infer_count_, _durations[StartPipeline].count());
  perf_counters_["4. output transfer from a device"] = makeProfileInfo(
      // Sum of all Results divided by count of infer requests
      result_timing == layer_timing.cend() ? 0 : result_timing->second * ms2us / infer_count_);
  perf_counters_["5. output postprocessing"] = makeProfileInfo(0, _durations[Postprocess].count());
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
    case Precision::I32:
        blob = InferenceEngine::make_shared_blob<std::int32_t>({Precision::I32, shape, layout});
        break;
    case Precision::U8:
        blob = InferenceEngine::make_shared_blob<uint8_t>({
                Precision::U8, shape, layout });
        break;
    case Precision::BOOL:
        blob = InferenceEngine::make_shared_blob<bool>({Precision::BOOL, shape, layout});
        break;
    default:
        throwIEException(fmt::format(
            "Cuda Plugin: Unsupported Input/Output Precision {}", precision));
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
    case Type_t::i32:
        return Precision::I32;
    case Type_t::boolean:
        return Precision::BOOL;
    default:
        throwIEException(
            fmt::format("Cuda Plugin: Unsupported Input/Output type {}", type));
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
    std::transform(
        begin,
        begin + src->size(),
        InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<float*>(),
        __half2float);
}

template<>
void CudaInferRequest::convertPrecision<float, __half>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const float*>();
    std::transform(
        begin,
        begin + src->size(),
        InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<__half*>(),
        __float2half);
}

template<>
void CudaInferRequest::convertPrecision<std::uint8_t, __half>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const std::uint8_t*>();
    std::transform(
        begin,
        begin + src->size(),
        InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<__half*>(),
        [](auto x) { return __float2half(static_cast<float>(x)); });
}

template<>
void CudaInferRequest::convertPrecision<__half, std::uint8_t>(const Blob::Ptr& src, const Blob::Ptr& dst) {
    auto begin = InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const __half*>();
    std::transform(
        begin,
        begin + src->size(),
        InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<std::uint8_t*>(),
        [](auto x) { return static_cast<std::uint8_t>(static_cast<float>(x)); });
}

void CudaInferRequest::convertPrecision(const Blob::Ptr& src, const Blob::Ptr& dst) {
    if (src->getTensorDesc().getPrecision() == dst->getTensorDesc().getPrecision()) {
        return;
    }
    switch (src->getTensorDesc().getPrecision()) {
        case Precision::U8 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP16:
                  convertPrecision<std::uint8_t, __half>(src, dst);
                  break;
                case Precision::FP32:
                    convertPrecision<std::uint8_t, float>(src, dst);
                    break;
                default : {
                    throwIEException(fmt::format(
                        "Unsupported precision conversion from {} to {}",
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
                    throwIEException(fmt::format(
                        "Unsupported precision conversion from {} to {}",
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
                default : {
                    throwIEException(fmt::format(
                        "Unsupported precision conversion from {} to {}",
                        src->getTensorDesc().getPrecision(),
                        dst->getTensorDesc().getPrecision()));
                }
            }
        } break;
        default : {
            throwIEException(
                fmt::format("Unsupported precision conversion from {}",
                            src->getTensorDesc().getPrecision()));
        }
    }
}

} // namespace CUDAPlugin
