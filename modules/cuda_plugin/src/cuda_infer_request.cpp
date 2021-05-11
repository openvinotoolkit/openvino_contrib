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

using Time = std::chrono::high_resolution_clock;

CudaInferRequest::CudaInferRequest(const InferenceEngine::InputsDataMap&                     networkInputs,
                                   const InferenceEngine::OutputsDataMap&                    networkOutputs,
                                   const std::shared_ptr<ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork) {
    // TODO: allocate infer request device and host buffers if needed, fill actual list of profiling tasks

    auto requestID = std::to_string(_executableNetwork->request_id_.fetch_add(1));

    std::string name = _executableNetwork->function_->get_friendly_name() + "_Req" + requestID;
    _profilingTask = {
        openvino::itt::handle("Cuda" + std::to_string(_executableNetwork->cfg_.deviceId) + "_" + name + "_Preprocess"),
        openvino::itt::handle("Cuda" + std::to_string(_executableNetwork->cfg_.deviceId) + "_" + name + "_Postprocess"),
        openvino::itt::handle("Cuda" + std::to_string(_executableNetwork->cfg_.deviceId) + "_" + name + "_StartPipline"),
        openvino::itt::handle("Cuda" + std::to_string(_executableNetwork->cfg_.deviceId) + "_" + name + "_WaitPipline"),
    };

    _executable = _executableNetwork->plugin_->_backend->compile(_executableNetwork->function_);
    _parameters = _executableNetwork->function_->get_parameters();
    _results = _executableNetwork->function_->get_results();

    allocateDeviceBuffers();
    allocateBlobs();
}

CudaInferRequest::~CudaInferRequest() {
    _executableNetwork->request_id_--;
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
    auto&& parameters = _executableNetwork->function_->get_parameters();
    AllocateImpl(_networkInputs, _inputs, _deviceInputs, [&] (const std::string& blobName) {
        return parameters.at(_executableNetwork->input_index_.at(blobName))->get_element_type();
    });
    auto&& results = _executableNetwork->function_->get_results();
    AllocateImpl(_networkOutputs, _outputs, _networkOutputBlobs, [&] (const std::string& blobName) {
        return results.at(_executableNetwork->output_index_.at(blobName))->get_element_type();
    });
}

void CudaInferRequest::InferImpl() {
    inferPreprocess();
    startPipeline();
    waitPipeline();  // does nothing in current implementation
    inferPostprocess();
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

void CudaInferRequest::setCudaStream(std::shared_ptr<CudaStream> cudaStream) {
    cuda_stream_ = std::move(cudaStream);
}

void CudaInferRequest::inferPreprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Preprocess]);
    auto start = Time::now();
    // NOTE: After InferRequestInternal::execDataPreprocessing call
    //       input can points to other memory region than it was allocated in constructor.
    InferRequestInternal::execDataPreprocessing(_deviceInputs);
    for (auto&& networkInput : _deviceInputs) {
        auto index = _executableNetwork->input_index_[networkInput.first];
        const auto& parameter = _parameters[index];
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        _inputTensors[index] = _executableNetwork->plugin_->_backend->create_tensor(
            parameterType,
            parameterShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput.second)->rmap().as<void*>());
    }
    for (auto&& output : _outputs) {
        auto outputBlob = output.second;
        auto networkOutput = _networkOutputBlobs[output.first];
        auto index = _executableNetwork->output_index_[output.first];
        if (outputBlob->getTensorDesc().getPrecision() == networkOutput->getTensorDesc().getPrecision()) {
            networkOutput = outputBlob;
        }
        const auto& result = _results[index];
        const auto& resultShape = result->get_shape();
        const auto& resultType = result->get_element_type();
        _outputTensors[index] = _executableNetwork->plugin_->_backend->create_tensor(
            resultType,
            resultShape,
            InferenceEngine::as<InferenceEngine::MemoryBlob>(networkOutput)->wmap().as<void*>());
    }
    _durations[Preprocess] = Time::now() - start;
}

void CudaInferRequest::startPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[StartPipeline])
    memory_manager_proxy_ = _executableNetwork->memory_manager_pool_->WaitAndGet();
    auto start = Time::now();
    auto inferRequestContext = InferenceRequestContext{cuda_stream_, {}, {}};
    for (auto& op : _executableNetwork->exec_sequence_) {
      auto inputTensors = memory_manager_proxy_->Get().inputTensorPointers(*op);
      auto outputTensors = memory_manager_proxy_->Get().outputTensorPointers(*op);
      op->Execute(inferRequestContext, inputTensors, outputTensors);
    }
    _durations[StartPipeline] = Time::now() - start;
}

void CudaInferRequest::waitPipeline() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[WaitPipeline])
    auto start = Time::now();
    cuda_stream_->synchronize();
    memory_manager_proxy_.reset();
    _durations[WaitPipeline] = Time::now() - start;
}

void CudaInferRequest::inferPostprocess() {
    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin, _profilingTask[Postprocess]);
    auto start = Time::now();
    for (auto&& output : _outputs) {
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
