// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_base.hpp>
#include <cuda_itopology_runner.hpp>
#include <memory_manager/cuda_memory_manager.hpp>
#include <memory_manager/cuda_memory_pool.hpp>

#include "openvino/op/util/sub_graph_base.hpp"

namespace ov {
namespace nvidia_gpu {

class SubGraph : public OperationBase {
public:
    using ExecSequence = std::vector<OperationBase::Ptr>;

    SubGraph(const CreationContext& context, const std::shared_ptr<const ov::Model>& model);

    SubGraph(const CreationContext& context,
             const std::shared_ptr<const ov::Model>& model,
             const ExecSequence& sequence,
             const std::shared_ptr<MemoryManager>& memoryManager);

    virtual ~SubGraph() = default;

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

    void Capture(InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    void ExecuteGraph(InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const override;

    virtual void initializeRunner();

    inline std::shared_ptr<MemoryManager> memoryManager() const { return memory_manager_; }

    inline const std::vector<OperationBase::Ptr>& getExecSequence() const { return exec_sequence_; }

    inline const std::shared_ptr<const ov::Model> getModel() const { return model_; };

    const std::vector<OperationBase::Ptr>& getParams() const;
    const std::vector<OperationBase::Ptr>& getResults() const;

    bool hasTopologyRunners() const {
        if (runners_status_ == NestedRunnersStatus::UNKNOWN) {
            if (runner_ != nullptr) {
                runners_status_ = NestedRunnersStatus::PRESENT;
            } else {
                runners_status_ = NestedRunnersStatus::ABSENT;
                for (const auto& op : exec_sequence_) {
                    const auto sg = std::dynamic_pointer_cast<SubGraph>(op);
                    if (sg && sg->hasTopologyRunners()) {
                        runners_status_ = NestedRunnersStatus::PRESENT;
                        break;
                    }
                }
            }
        }
        return runners_status_ == NestedRunnersStatus::PRESENT;
    }

    virtual std::size_t GetCudaGraphsCount() const;

private:
    void initSharedImmutableWorkbuffers(const std::vector<OperationBase::Ptr>& init_sequence);
    void initExecuteSequence(bool isStableParams, bool isStableResults);
    static std::unique_ptr<MemoryManager> createMemoryManager(const OperationBuffersExtractor& opBuffersExtractor);
    std::vector<DevicePointer<void*>> getSharedWorkbuffers(const IOperationExec& operation);

protected:
    enum class NestedRunnersStatus { UNKNOWN = -1, ABSENT, PRESENT };

    using SubGraphOp = ov::op::util::SubGraphOp;

    SubGraph(const CreationContext& context,
             const SubGraphOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);

    WorkbufferRequest GetWorkBufferRequest() const override;

    template <typename TNode>
    std::size_t getTensorByteSize(const TNode& node) {
        return node.get_element_type().size() * shape_size(node.get_shape());
    }

    struct OperationInfo {
        OperationInfo() = default;
        OperationInfo(const std::size_t size, const ov::element::Type type, ov::Shape shape)
            : size_{size}, type_{type}, shape_{std::move(shape)} {}
        std::size_t size_{};
        ov::element::Type type_{};
        ov::Shape shape_{};
    };

    std::shared_ptr<MemoryManager> memory_manager_;
    std::vector<OperationBase::Ptr> params_;
    std::vector<OperationInfo> params_info_;
    std::vector<OperationBase::Ptr> exec_sequence_;
    std::vector<OperationBase::Ptr> results_;
    std::vector<OperationInfo> results_info_;
    std::shared_ptr<const ov::Model> model_;

    const CreationContext& creation_context_;
    std::shared_ptr<ITopologyRunner> runner_ = nullptr;

    mutable CudaGraphCompatibility graph_compatibility_;
    mutable bool is_compatibility_analyzed_ = false;
    mutable NestedRunnersStatus runners_status_{NestedRunnersStatus::UNKNOWN};
};

}  // namespace nvidia_gpu
}  // namespace ov
