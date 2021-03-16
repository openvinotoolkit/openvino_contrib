// Copyright (C) 2020-2021 Intel Corporation
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
#include <ngraph/function.hpp>

#include "arm_infer_request.hpp"
#include "arm_executable_network.hpp"
#include "arm_plugin.hpp"

#include <half/half.hpp>

using namespace ArmPlugin;
using namespace InferenceEngine;
using namespace openvino;

using Time = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;
using fsec = std::chrono::duration<float>;

ArmInferRequest::ArmInferRequest(const InferenceEngine::InputsDataMap&                networkInputs,
                                 const InferenceEngine::OutputsDataMap&               networkOutputs,
                                 const std::shared_ptr<ArmPlugin::ExecutableNetwork>& executableNetwork) :
    InferRequestInternal(networkInputs, networkOutputs),
    _executableNetwork(executableNetwork),
    _lifetime{std::make_shared<arm_compute::OffsetLifetimeManager>()},
    _pool{std::make_shared<arm_compute::PoolManager>()},
    _memoryManager{std::make_shared<arm_compute::MemoryManagerOnDemand>(_lifetime, _pool)},
    _memoryGroup{_memoryManager} {
    auto requestID = std::to_string(_executableNetwork->_requestId.fetch_add(1));
    _profilingTasks =  {
        openvino::itt::handle("Arm_Preproc_" + _executableNetwork->_function->get_friendly_name() + "_" + requestID),
        openvino::itt::handle("Arm_Run_" + _executableNetwork->_function->get_friendly_name() + "_" + requestID),
        openvino::itt::handle("Arm_Postproc_" + _executableNetwork->_function->get_friendly_name() + "_" + requestID),
    };
    Layer::Map layers;
    IE_ASSERT(_executableNetwork->_executor != nullptr);
    _executableNetwork->_executor->runAndWait({
        [&] {
            layers = Converter{_executableNetwork->_function}.Configure(_memoryManager, _memoryGroup);
        }
    });
    for (auto&& node : _executableNetwork->_function->get_parameters()) {
        for (auto&& output : node->outputs()) {
            auto& tensor = layers.at(node->get_friendly_name())._outputs.at(output.get_index());
            if (tensor->info()->has_padding()) {
                tensor->allocator()->allocate();
            }
            _inputTensors.emplace(node->get_friendly_name(), tensor.get());
        }
    }
    for (auto&& node : _executableNetwork->_function->get_results()) {
        for (auto&& input : node->inputs()) {
            auto& tensor = layers.at(node->get_friendly_name())._inputs.at(input.get_index());
            if (tensor->info()->has_padding()) {
                tensor->allocator()->allocate();
            }
            auto sourceOutput = input.get_source_output();
            auto outputName = sourceOutput.get_node()->get_friendly_name();
            if (sourceOutput.get_node()->get_output_size() > 1) {
                outputName += '.' + std::to_string(sourceOutput.get_index());
            }
            _outputTensors.emplace(outputName, tensor);
        }
    }
    _memoryManager->populate(_allocator, 1);
    for (auto&& node : _executableNetwork->_function->get_ordered_ops()) {
        _layers.emplace_back(std::move(layers.at(node->get_friendly_name())));
        _layerNames.emplace_back(node->get_friendly_name());
        _layerTypes.emplace(node->get_friendly_name(), std::string {node->get_type_name()} + '.' + std::to_string(node->get_type_info().version));
    }
    allocateBlobs();
}

ArmInferRequest::~ArmInferRequest() {
    _executableNetwork->_requestId--;
}

void ArmInferRequest::allocateBlobs() {
    auto allocateImpl = [] (auto&& blobDataMap, auto& blobMap, auto& networkBlobMap, auto&& tensors) {
        for (auto&& blobData : blobDataMap) {
            auto& inferRequestBlob = blobMap[blobData.first];
            inferRequestBlob = make_blob_with_precision(blobData.second->getTensorDesc());
            inferRequestBlob->allocate();
            auto& precision = blobData.second->getTensorDesc().getPrecision();
            auto dataTypeCast = [] (const arm_compute::DataType type) {
                switch (type) {
                    case arm_compute::DataType::U8          : return InferenceEngine::Precision::U8;
                    case arm_compute::DataType::S16         : return InferenceEngine::Precision::I16;
                    case arm_compute::DataType::U16         : return InferenceEngine::Precision::U16;
                    case arm_compute::DataType::S32         : return InferenceEngine::Precision::I32;
                    case arm_compute::DataType::S64         : return InferenceEngine::Precision::U32;
                    case arm_compute::DataType::F16         : return InferenceEngine::Precision::FP16;
                    case arm_compute::DataType::F32         : return InferenceEngine::Precision::FP32;
                    case arm_compute::DataType::BFLOAT16    : return InferenceEngine::Precision::BF16;
                    default: THROW_IE_EXCEPTION << "Unsupported Data Type ";
                }
            };
            auto networkPresion = dataTypeCast(tensors.at(blobData.first)->info()->data_type());
            auto& networkBlob = networkBlobMap[blobData.first];
            if (networkPresion == precision) {
                networkBlob = inferRequestBlob;
            } else {
                auto& dims = blobData.second->getTensorDesc().getDims();
                auto layout = blobData.second->getTensorDesc().getLayout();
                networkBlob = make_blob_with_precision(TensorDesc{networkPresion, dims, layout});
            }
            if (inferRequestBlob != networkBlob) {
                networkBlob->allocate();
            }
        }
    };
    allocateImpl(_networkInputs, _inputs, _networkInputBlobs, _inputTensors);
    allocateImpl(_networkOutputs, _outputs, _networkOutputBlobs, _outputTensors);
}

template<typename SrcT, typename DstT>
static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
    std::copy_n(InferenceEngine::as<InferenceEngine::MemoryBlob>(src)->rmap().as<const SrcT*>(),
                src->size(),
                InferenceEngine::as<InferenceEngine::MemoryBlob>(dst)->wmap().as<DstT*>());
}

static void blobCopy(const Blob::Ptr& src, const Blob::Ptr& dst) {
#define DST_CASE(srcT, dstP, dstT)    \
    case Precision::dstP : blobCopy<srcT, dstT>(src, dst); break;
#define SRC_CASE(srcP, srcT)                                                \
   case Precision::srcP : switch (dst->getTensorDesc().getPrecision()) {    \
        DST_CASE(srcT, U8,    std::uint8_t)                                 \
        DST_CASE(srcT, I16,   std::int16_t)                                 \
        DST_CASE(srcT, U16,   std::uint16_t)                                \
        DST_CASE(srcT, I32,   std::int32_t)                                 \
        DST_CASE(srcT, U32,   std::uint32_t)                                \
        DST_CASE(srcT, I64,   std::int64_t)                                 \
        DST_CASE(srcT, U64,   std::uint64_t)                                \
        DST_CASE(srcT, FP16,  half_float::half)                             \
        DST_CASE(srcT, FP32,  float)                                        \
        default: THROW_IE_EXCEPTION << "Unsupported Data Type " << dst->getTensorDesc().getPrecision();            \
    } break;
    switch (src->getTensorDesc().getPrecision()) {
        SRC_CASE(BOOL,  bool)
        SRC_CASE(U8,    std::uint8_t)
        SRC_CASE(I16,   std::int16_t)
        SRC_CASE(U16,   std::uint16_t)
        SRC_CASE(I32,   std::int32_t)
        SRC_CASE(U32,   std::uint32_t)
        SRC_CASE(I64,   std::int64_t)
        SRC_CASE(U64,   std::uint64_t)
        SRC_CASE(FP16,  half_float::half)
        SRC_CASE(FP32,  float)
        default: THROW_IE_EXCEPTION << "Unsupported Data Type " << src->getTensorDesc().getPrecision();
    }
#undef SRC_CASE
#undef DST_CASE
}

void ArmInferRequest::InferImpl() {
    arm_compute::MemoryGroupResourceScope memoryGroupScope(_memoryGroup);
    {
        OV_ITT_SCOPED_TASK(Itt::Domains::ArmPlugin, _profilingTasks[Preprocessing]);
        auto start = Time::now();
        InferRequestInternal::execDataPreprocessing(_inputs);
        for (auto&& input : _inputs) {
            auto inputBlob = input.second;
            auto networkInput = _networkInputBlobs[input.first];
            if (inputBlob->getTensorDesc().getPrecision() == networkInput->getTensorDesc().getPrecision()) {
                networkInput = inputBlob;
            } else {
                blobCopy(inputBlob, networkInput);
            }
            auto inputTensor = _inputTensors[input.first];
            if (inputTensor->info()->has_padding()) {
                arm_compute::Tensor networkInputTensor;
                networkInputTensor.allocator()->init({inputTensor->info()->tensor_shape(), 1, inputTensor->info()->data_type()});
                networkInputTensor.allocator()->import_memory(
                    InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput)->rmap().as<void*>());
                inputTensor->copy_from(networkInputTensor);
            } else {
                static_cast<arm_compute::Tensor*>(inputTensor)->allocator()->import_memory(
                    InferenceEngine::as<InferenceEngine::MemoryBlob>(networkInput)->rmap().as<void*>());
            }
        }
        for (auto output : _outputTensors) {
            auto outputTensor = output.second;
            if (_networkOutputBlobs.find(output.first) == _networkOutputBlobs.end()) {
                if (!outputTensor->info()->has_padding()) {
                    static_cast<arm_compute::Tensor*>(outputTensor)->allocator()->allocate();
                }
            } else {
                auto networkOutput = _networkOutputBlobs[output.first];
                auto outputBlob = _outputs[output.first];
                if (outputBlob->getTensorDesc().getPrecision() == networkOutput->getTensorDesc().getPrecision()) {
                    networkOutput = outputBlob;
                }
                if (!outputTensor->info()->has_padding() && _layerTypes[output.first] != "Constant.0") {
                    static_cast<arm_compute::Tensor*>(outputTensor)->allocator()->import_memory(
                        InferenceEngine::as<InferenceEngine::MemoryBlob>(networkOutput)->wmap().as<void*>());
                }
            }
        }
        _durations["Preprocessing"] = Time::now() - start;
    }
    {
        OV_ITT_SCOPED_TASK(Itt::Domains::ArmPlugin, _profilingTasks[Run]);
        for (std::size_t i = 0; i < _layers.size(); ++i) {
            if (_layers[i]._function != nullptr) {
                auto start = Time::now();
                _layers[i]._function->run();
                _durations[_layerNames[i]] = Time::now() - start;
            }
        }
    }
    {
        OV_ITT_SCOPED_TASK(Itt::Domains::ArmPlugin, _profilingTasks[Postprocessing]);
        auto start = Time::now();
        for (auto&& output : _outputs) {
            auto networkOutput = _networkOutputBlobs[output.first];
            auto outputTensor = _outputTensors[output.first];
            if (outputTensor->info()->has_padding() || _layerTypes[output.first] == "Constant.0") {
                arm_compute::Tensor networkOutputTensor;
                networkOutputTensor.allocator()->init({outputTensor->info()->tensor_shape(), 1, outputTensor->info()->data_type()});
                networkOutputTensor.allocator()->import_memory(
                    InferenceEngine::as<InferenceEngine::MemoryBlob>(networkOutput)->wmap().as<void*>());
                networkOutputTensor.copy_from(*outputTensor);
            }
            auto outputBlob = output.second;
            if (outputBlob->getTensorDesc().getPrecision() != networkOutput->getTensorDesc().getPrecision() ||
                _layerTypes[output.first] == "Constant.0") {
                blobCopy(networkOutput, outputBlob);
            }
        }
        _durations["Postprocessing"] = Time::now() - start;
    }
}

std::map<std::string, InferenceEngineProfileInfo> ArmInferRequest::GetPerformanceCounts() const {
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    InferenceEngineProfileInfo info;
    info.execution_index = 0;
    info.status = InferenceEngineProfileInfo::EXECUTED;
    info.cpu_uSec = 0;
    for (auto&& value : _durations) {
        info.realTime_uSec = value.second.count();
        auto itType = _layerTypes.find(value.first);
        if (itType != _layerTypes.end()) {
            auto& layerType = itType->second;
            auto pos = std::copy_n(layerType.c_str(), std::min(sizeof(info.layer_type) - 1, layerType.size()), info.layer_type);
            *pos = '\0';
        } else {
            info.layer_type[0] = '\0';
        }
        perfMap[value.first] = info;
    }
    return perfMap;
}
