// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cuda_operation_base.hpp>
#include <cuda/graph.hpp>
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

    CudaGraphCompatibility GetCudaGraphCompatibility() const override;

    void Capture(InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {
        if (hasTopologyRunners()) {
            CaptureMulti(context, inputTensors, outputTensors, workbuffers);
        } else {
            CaptureSingle(context, inputTensors, outputTensors, workbuffers);
        }
    }

    void ExecuteGraph(InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const override {
        if (hasTopologyRunners()) {
            ExecuteGraphMulti(context, inputTensors, outputTensors, workbuffers);
        } else {
            ExecuteGraphSingle(context, inputTensors, outputTensors, workbuffers);
        }
    }

    void initializeRunner() override;

    std::size_t GetCudaGraphsCount() const override {
        if (hasTopologyRunners()) {
            return 3;
        }
        return 1;
    }

private:
    struct PortMap {
        int64_t start{0};
        int64_t stride{0};
        int64_t part_size{0};
        int64_t end{0};
        int64_t axis{0};
    };

    class SliceLauncher {
    public:
        SliceLauncher(const TensorIteratorOp& ti, uint64_t inputIdx, uint64_t paramIdx);

        void operator()(const CUDA::Stream& stream,
                        const IOperationExec::Inputs& inputTensors,
                        CUDA::DevicePointer<void*> mutableBuffer,
                        int64_t iter) const {
            const auto* src = inputTensors[input_idx_].get();
            auto* dst = memory_manager_.outputTensorPointers(param_, mutableBuffer)[0].get();
            slice_(stream.get(), src, dst, start_ + iter * stride_);
        }

        void addKernelNode(ICudaGraphInfo& info,
                           const CUDA::Stream& stream,
                           CUDA::DevicePointer<void*> mutableBuffer,
                           const IOperationExec::Inputs& inputTensors);

        void updateKernelNode(ICudaGraphInfo& info,
                              std::size_t index,
                              CUDA::DevicePointer<void*> mutableBuffer,
                              const IOperationExec::Inputs& inputTensors,
                              int64_t iter) {
            const auto* src = inputTensors[input_idx_].get();
            auto* dst = memory_manager_.outputTensorPointers(param_, mutableBuffer)[0].get();
            info.update_kernel(index, slice_.getPropsPtr(), start_ + iter * stride_, slice_.getSize(), src, dst);
        }

    private:
        uint64_t input_idx_;
        const OperationBase& param_;
        const MemoryManager& memory_manager_;
        const kernel::Slice& slice_;
        size_t start_;
        int64_t stride_;
    };

    class TransferLauncher {
    public:
        TransferLauncher(const TensorIteratorOp& ti, uint64_t resultIdx, uint64_t paramIdx);

        void operator()(const CUDA::Stream& stream, CUDA::DevicePointer<void*> mutableBuffer) const {
            const auto& paramTensors = memory_manager_.outputTensorPointers(param_, mutableBuffer);
            const auto& resultTensors = memory_manager_.inputTensorPointers(result_, mutableBuffer);
            auto* dst = paramTensors[0].get();
            const auto* src = resultTensors[0].get();
            throwIfError(cudaMemcpyAsync(dst, src, param_size_, cudaMemcpyDeviceToDevice, stream.get()));
        }

        void addTransferNode(ICudaGraphInfo& info,
                             const CUDA::Stream& stream,
                             CUDA::DevicePointer<void*> mutableBuffer);

    private:
        const OperationBase& param_;
        const OperationBase& result_;
        const MemoryManager& memory_manager_;
        std::size_t param_size_;
    };

    class InsertLauncher {
    public:
        InsertLauncher(const TensorIteratorOp& ti, const std::size_t resultIdx, const std::size_t outputIdx);

        void operator()(const CUDA::Stream& stream,
                        CUDA::DevicePointer<void*> mutableBuffer,
                        const IOperationExec::Outputs& outputTensors,
                        int64_t iter) const {
            const auto* src = memory_manager_.inputTensorPointers(result_, mutableBuffer)[0].get();
            auto* dst = outputTensors[output_idx_].get();
            insert_(stream.get(), src, dst, start_ + iter * stride_);
        }

        void addKernelNode(ICudaGraphInfo& info,
                           const CUDA::Stream& stream,
                           CUDA::DevicePointer<void*> mutableBuffer,
                           const IOperationExec::Outputs& outputTensors);

        void updateKernelNode(ICudaGraphInfo& info,
                              std::size_t index,
                              CUDA::DevicePointer<void*> mutableBuffer,
                              const IOperationExec::Outputs& outputTensors,
                              int64_t iter) {
            const auto* src = memory_manager_.inputTensorPointers(result_, mutableBuffer)[0].get();
            auto* dst = outputTensors[output_idx_].get();
            info.update_kernel(index, insert_.getPropsPtr(), start_ + iter * stride_, insert_.getSize(), src, dst);
        }

    private:
        uint64_t output_idx_;
        const OperationBase& result_;
        const MemoryManager& memory_manager_;
        size_t start_;
        int64_t stride_;
        const kernel::Insert& insert_;
    };

    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers& buffers) override;

    void CaptureSingle(InferenceRequestContext& context,
                       Inputs inputTensors,
                       Outputs outputTensors,
                       const Workbuffers& workbuffers) const;

    void ExecuteGraphSingle(InferenceRequestContext& context,
                            Inputs inputTensors,
                            Outputs outputTensors,
                            const Workbuffers& workbuffers) const;

    void CaptureMulti(InferenceRequestContext& context,
                      Inputs inputTensors,
                      Outputs outputTensors,
                      const Workbuffers& workbuffers) const;

    void ExecuteGraphMulti(InferenceRequestContext& context,
                           Inputs inputTensors,
                           Outputs outputTensors,
                           const Workbuffers& workbuffers) const;

    void transferParam(const CUDA::Stream& stream,
                       CUDA::DevicePointer<void*> mutableBuffer,
                       const IOperationExec::Inputs& inputTensors,
                       std::int64_t iter,
                       uint64_t inputIdx,
                       uint64_t paramIdx) const;

    void transferResult(const CUDA::Stream& stream,
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

    mutable std::vector<SliceLauncher> slices_;
    mutable std::vector<TransferLauncher> transfers_;
    mutable std::vector<InsertLauncher> inserts_;
};

}  // namespace nvidia_gpu
}  // namespace ov
