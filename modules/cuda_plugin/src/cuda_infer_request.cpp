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

using namespace InferenceEngine;

namespace CUDAPlugin {

using Time = std::chrono::steady_clock;

CudaInferRequest::CudaInferRequest(const InferenceEngine::InputsDataMap&                     networkInputs,
                                   const InferenceEngine::OutputsDataMap&                    networkOutputs,
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

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

void CudaInferRequest::allocateDeviceBuffers() {
    // Allocate plugin backend specific memory handles
    _inputTensors.resize(_networkInputs.size());
    _outputTensors.resize(_networkOutputs.size());
}

template<typename BlobDataMap, typename GetNetworkPrecisionF>
static void AllocateImpl(const BlobDataMap& userDataMap,
                         BlobMap& userBlobMap,
                         BlobMap& deviceBlobMap,
                         GetNetworkPrecisionF&& GetNetworkPrecision) {
    for (auto&& userData : userDataMap) {
        auto& dims = userData.second->getTensorDesc().getDims();
        const auto devicePrecision = Precision::FP32;
        const auto deviceLayout = TensorDesc::getLayoutByDims(dims);
        auto userPrecision = userData.second->getTensorDesc().getPrecision();
        auto userLayout = userData.second->getTensorDesc().getLayout();

        Blob::Ptr userBlob;
        switch (userPrecision) {
            case Precision::U8: {
                userBlob = InferenceEngine::make_shared_blob<std::uint8_t>({userPrecision, dims, userLayout});
            } break;
            case Precision::FP32 : {
                userBlob = InferenceEngine::make_shared_blob<float>({userPrecision, dims, userLayout});
            } break;
            default: //IE_THROW() << "Cuda Plugin: Unsupported Input/Output Precision";
                     THROW_IE_EXCEPTION << "Cuda Plugin: Unsupported Input/Output Presision";
        }
        userBlob->allocate();
        userBlobMap[userData.first] = userBlob;

        auto networkPrecision = GetNetworkPrecision(userData.first);
        Blob::Ptr deviceBlob;
        switch (networkPrecision) {
            case ngraph::element::Type_t::f32 : {
                if (userPrecision == devicePrecision && userLayout == deviceLayout) {
                    deviceBlob = userBlob;
                } else {
                    deviceBlob = InferenceEngine::make_shared_blob<float>({devicePrecision, dims, deviceLayout});
                }
            } break;
            default: //IE_THROW() << "Cuda Plugin: Unsupported network Input/Output Presision";
                     THROW_IE_EXCEPTION << "Cuda Plugin: Unsupported network Input/Output Presision";
        }
        // preprocessing converts user input blob to desired device input blob automatically
        // NOTE: this is not supported for output user blobs yet
        if (userBlob != deviceBlob) {
            deviceBlob->allocate();
        }
        deviceBlobMap[userData.first] = deviceBlob;
    }
}

void CudaInferRequest::allocateBlobs() {
    AllocateImpl(_networkInputs, _inputs, _deviceInputs, [&] (const std::string& blobName) {
        return _executableNetwork->parameter(blobName).get_element_type();
    });
    AllocateImpl(_networkOutputs, _outputs, _networkOutputBlobs, [&] (const std::string& blobName) {
        return _executableNetwork->result(blobName).get_element_type();
    });
}

template<typename SrcT, typename DstT>
static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    std::copy_n(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                src->size(),
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>());
}

static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    switch (src->getTensorDesc().getPrecision()) {
        case Precision::U8 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::U8 : break;
                case Precision::FP32 : {
                    blobCopy<std::uint8_t, float>(src, dst);
                } break;
                default : {
                    //IE_THROW() << "Unsupported precision conversion from "
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        case Precision::FP32 : {
            switch (dst->getTensorDesc().getPrecision()) {
                case Precision::FP32 : break;
                case Precision::U8 : {
                    blobCopy<float, std::uint8_t>(src, dst);
                } break;
                default : {
                    // IE_THROW() << "Unsupported precision conversion from "
                    THROW_IE_EXCEPTION << "Unsupported precision conversion from "
                        << src->getTensorDesc().getPrecision() <<" to " << dst->getTensorDesc().getPrecision();
                }
            }
        } break;
        default : {
            //IE_THROW() << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision();
            THROW_IE_EXCEPTION << "Unsupported precision conversion from " << src->getTensorDesc().getPrecision();
        }
    }
}

void CudaInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Preprocess]);
    ThrowIfCanceled();
    auto start = Time::now();
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    InferRequestInternal::execDataPreprocessing(_deviceInputs);
    ThrowIfCanceled();
    _executableNetwork->createInputs(_inputTensors, _deviceInputs);
    ThrowIfCanceled();
    _executableNetwork->createOutputs(_outputTensors, _outputs, _networkOutputBlobs);
    ThrowIfCanceled();
    _durations[Preprocess] = Time::now() - start;
}

void CudaInferRequest::startPipeline(const CUDA::ThreadContext& threadContext) {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[StartPipeline])
    auto start = Time::now();
    ThrowIfCanceled();
    memory_manager_proxy_ =
        _executableNetwork->memory_manager_pool_->WaitAndGet();
    ThrowIfCanceled();
    auto& manager = memory_manager_proxy_->Get();
    InferenceRequestContext inferRequestContext{_inputs, _outputs, threadContext};
    for (auto& op : _executableNetwork->exec_sequence_) {
      ThrowIfCanceled();
      auto inputTensors = manager.inputTensorPointers(*op);
      auto outputTensors = manager.outputTensorPointers(*op);
      op->Execute(inferRequestContext, inputTensors, outputTensors);
    }
    _durations[StartPipeline] = Time::now() - start;
}

void CudaInferRequest::waitPipeline(const CUDA::ThreadContext& threadContext) {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[WaitPipeline])
    ThrowIfCanceled();
    auto start = Time::now();
    // TODO: probably all time will be spent in synchonize, out of reach of ThrowIfCanceled
    threadContext.stream().synchronize();
    memory_manager_proxy_.reset();
    _durations[WaitPipeline] = Time::now() - start;
}

void CudaInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Postprocess]);
    ThrowIfCanceled();
    auto start = Time::now();
    for (auto&& output : _outputs) {
        ThrowIfCanceled();
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        // perform precision conversion of network output's precision and computational
        // graph output's precision are different
        if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision()) {
            blobCopy(networkOutput, outputBlob);
        }
    }
    _durations[Postprocess] = Time::now() - start;
}

void CudaInferRequest::Cancel() {
    cancellation_token_.store(true, std::memory_order_release);
}

void CudaInferRequest::ThrowIfCanceled() {
    if (cancellation_token_.load(std::memory_order_acquire)) {
        memory_manager_proxy_.reset();
        cancellation_token_.store(false, std::memory_order_release);
        THROW_IE_EXCEPTION_WITH_STATUS(INFER_CANCELLED);
    }
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

} // namespace CUDAPlugin
