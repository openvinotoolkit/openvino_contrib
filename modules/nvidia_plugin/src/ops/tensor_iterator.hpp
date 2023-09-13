// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cuda_operation_base.hpp>
#include <kernels/insert.hpp>
#include <kernels/slice.hpp>
#include <openvino/op/tensor_iterator.hpp>

#include "subgraph.hpp"

namespace ov {
namespace nvidia_gpu {

class TensorIteratorOp : public SubGraph {
public:
    using NodeOp = ov::op::v0::TensorIterator;
    TensorIteratorOp(const CreationContext& context,
                     const NodeOp& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    bool IsCudaGraphCompatible() const override;

    void Capture(InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    struct PortMap {
        int64_t start{0};
        int64_t stride{0};
        int64_t part_size{0};
        int64_t end{0};
        int64_t axis{0};
    };

    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

    void copyParam(const CUDA::Stream& stream,
                   CUDA::DevicePointer<void*> mutableBuffer,
                   const IOperationExec::Inputs& inputTensors,
                   std::int64_t iter,
                   uint64_t inputIdx,
                   uint64_t paramIdx) const;
    void copyBackEdge(const CUDA::Stream& stream,
                      CUDA::DevicePointer<void*> mutableBuffer,
                      uint64_t resultIdx,
                      uint64_t paramIdx) const;
    void copyResult(const CUDA::Stream& stream,
                    CUDA::DevicePointer<void*> mutableBuffer,
                    const IOperationExec::Outputs& outputTensors,
                    int64_t iter,
                    std::size_t resultIdx,
                    std::size_t outputIdx) const;

    void updateExecSequence();

    size_t max_threads_per_block_;
    const int64_t num_iterations_;
    std::vector<OperationInfo> inputs_info_;
    std::vector<OperationInfo> outputs_info_;
    std::unordered_map<uint64_t, uint64_t> inputs_parameters_map_;
    std::vector<uint64_t> invariant_inputs_;
    std::unordered_map<uint64_t, PortMap> portmap_inputs_;
    std::unordered_map<uint64_t, kernel::Slice> kernelmap_inputs_;
    std::unordered_map<uint64_t, uint64_t> results_outputs_map_;
    std::unordered_map<uint64_t, std::vector<uint64_t>> iterations_results_map_;
    std::unordered_map<uint64_t, PortMap> portmap_outputs_;
    std::unordered_map<uint64_t, kernel::Insert> kernelmap_outputs_;
    std::unordered_map<uint64_t, uint64_t> results_parameters_map_;
};

}  // namespace nvidia_gpu
}  // namespace ov
