// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_base.hpp>
#include <memory_manager/cuda_memory_manager.hpp>
#include <memory_manager/cuda_memory_pool.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>

class ExecNetworkTest;

namespace CUDAPlugin {

class SubGraph : public OperationBase {
public:
    friend class ::ExecNetworkTest;

    virtual ~SubGraph() = 0;

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;
    const MemoryManager& memoryManager() const { return *memory_manager_; }

    const std::vector<OperationBase::Ptr>& getParams() const;
    const std::vector<OperationBase::Ptr>& getExecSequence() const;
    const std::vector<OperationBase::Ptr>& getResults() const;

private:
    void initSharedImmutableWorkbuffers(const std::vector<OperationBase::Ptr>& init_sequence);
    void initExecuteSequence(const CreationContext& context, bool isStableParams);
    static std::unique_ptr<MemoryManager> createMemoryManager(const OperationBuffersExtractor& opBuffersExtractor);
    std::vector<DevicePointer<void*>> getSharedWorkbuffers(const IOperationExec& operation);

protected:
    using SubGraphOp = ngraph::op::util::SubGraphOp;

    SubGraph(const CreationContext& context,
             const SubGraphOp& node,
             IndexCollection&& inputIds,
             IndexCollection&& outputIds);
    SubGraph(const CreationContext& context, const std::shared_ptr<const ngraph::Function>& function);

    WorkbufferRequest GetWorkBufferRequest() const override;

    template <typename TNode>
    std::size_t getTensorByteSize(const TNode& node) {
        return node.get_element_type().size() * shape_size(node.get_shape());
    }

    struct OperationInfo {
        OperationInfo() = default;
        OperationInfo(const std::size_t size, const ngraph::element::Type type, ngraph::Shape shape)
            : size_{size}, type_{type}, shape_{std::move(shape)} {}
        std::size_t size_{};
        ngraph::element::Type type_{};
        ngraph::Shape shape_{};
    };

    std::unique_ptr<MemoryManager> memory_manager_;
    std::vector<OperationBase::Ptr> params_;
    std::vector<OperationInfo> params_info_;
    std::vector<OperationBase::Ptr> exec_sequence_;
    std::vector<OperationBase::Ptr> results_;
    std::vector<OperationInfo> results_info_;
    std::shared_ptr<const ngraph::Function> function_;
};

inline SubGraph::~SubGraph() {}

inline const std::vector<OperationBase::Ptr>& SubGraph::getParams() const { return params_; }

inline const std::vector<OperationBase::Ptr>& SubGraph::getExecSequence() const { return exec_sequence_; }

inline const std::vector<OperationBase::Ptr>& SubGraph::getResults() const { return results_; }

}  // namespace CUDAPlugin
