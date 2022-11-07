// Copyright (C) 2020-2022 Intel Corporation
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

#include <arm_compute/runtime/BlobLifetimeManager.h>

namespace ArmPlugin {

class ExecutableNetwork;

struct ArmInferRequest : public InferenceEngine::IInferRequestInternal {
    using Ptr = std::shared_ptr<ArmInferRequest>;

    ArmInferRequest(const InferenceEngine::InputsDataMap&     networkInputs,
                    const InferenceEngine::OutputsDataMap&    networkOutputs,
                    const std::shared_ptr<ExecutableNetwork>& executableNetwork);
    ArmInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& networkInputs,
                    const std::vector<std::shared_ptr<const ov::Node>>& networkOutputs,
                    const std::shared_ptr<ExecutableNetwork>& executableNetwork);
    ~ArmInferRequest();

    void InferImpl() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    using Duration = std::chrono::duration<float, std::micro>;
    struct LayerInfo {
        Layer                   _layer;
        ngraph::Node*           _node;
        openvino::itt::handle_t _profilingTask;
        std::string             _execType;
        Duration                _duration;
        std::size_t             _counter;
    };
    struct IOInfo {
        Output                              _output;
        arm_compute::ITensor*               _tensor;
        openvino::itt::handle_t             _profilingTask;
        InferenceEngine::Blob::Ptr          _blob;
        InferenceEngine::BlobMap::iterator  _itBlob;
        std::string                         _execType;
        Duration                            _duration;
        std::size_t                         _counter;
    };
    std::shared_ptr<ExecutableNetwork>                                          _executableNetwork;
    std::vector<LayerInfo>                                                      _layers;
    std::vector<IOInfo>                                                         _inputInfo;
    std::vector<IOInfo>                                                         _outputInfo;
    arm_compute::Allocator                                                      _allocator;
    std::shared_ptr<arm_compute::ISimpleLifetimeManager>                        _lifetime;
    std::shared_ptr<arm_compute::PoolManager>                                   _pool;
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>                         _memoryManager;
    std::unique_ptr<arm_compute::MemoryGroup>                                   _memoryGroup;
    std::unique_ptr<arm_compute::MemoryGroupResourceScope>                      _memoryGroupScope;

private:
    void InitArmInferRequest(const std::shared_ptr<ArmPlugin::ExecutableNetwork>& executableNetwork);
};
// ! [infer_request:header]

}  // namespace ArmPlugin
