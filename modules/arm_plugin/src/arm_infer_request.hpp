// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include <ie_common.h>
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"
#include <threading/ie_itask_executor.hpp>

#include "arm_converter/arm_converter.hpp"
#include "arm_config.hpp"
#include "arm_itt.hpp"

#include <arm_compute/runtime/Allocator.h>
#include <arm_compute/runtime/OffsetLifetimeManager.h>
#include <arm_compute/runtime/PoolManager.h>
#include <arm_compute/runtime/MemoryManagerOnDemand.h>
#include <arm_compute/runtime/MemoryGroup.h>

namespace ArmPlugin {

class ExecutableNetwork;

struct ArmInferRequest : public InferenceEngine::IInferRequestInternal {
    using Ptr = std::shared_ptr<ArmInferRequest>;

    ArmInferRequest(const InferenceEngine::InputsDataMap&     networkInputs,
                      const InferenceEngine::OutputsDataMap&    networkOutputs,
                      const std::shared_ptr<ExecutableNetwork>& executableNetwork);
    virtual ~ArmInferRequest();

    void InferImpl() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void allocateBlobs();

    enum {Preprocessing, Run, Postprocessing, numOfStages};
    std::shared_ptr<ExecutableNetwork>                                          _executableNetwork;
    std::array<openvino::itt::handle_t, numOfStages>                            _profilingTasks;
    std::unordered_map<std::string, std::chrono::duration<float, std::micro>>   _durations;
    std::vector<Layer>                                                          _layers;
    std::vector<std::string>                                                    _layerNames;
    std::map<std::string, std::string>                                          _layerTypes;
    InferenceEngine::BlobMap                                                    _networkInputBlobs;
    InferenceEngine::BlobMap                                                    _networkOutputBlobs;
    std::unordered_map<std::string, arm_compute::ITensor*>                      _inputTensors;
    std::unordered_map<std::string, arm_compute::ITensor*>                      _outputTensors;
    arm_compute::Allocator                                                      _allocator;
    std::shared_ptr<arm_compute::OffsetLifetimeManager>                         _lifetime;
    std::shared_ptr<arm_compute::PoolManager>                                   _pool;
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>                         _memoryManager;
    arm_compute::MemoryGroup                                                    _memoryGroup;
};
// ! [infer_request:header]

}  // namespace ArmPlugin
