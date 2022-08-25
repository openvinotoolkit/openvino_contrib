// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <gsl/span>
#include <ops/components/workbuffer_desc.hpp>

namespace CUDAPlugin::RNN::Details {

/**
 * Base class for `TransposeInputTensorAdapter` and `TransposeOutputTensorAdapter`
 *
 * TODO: Consider to refactor using `TransposeOp` using aggregation or by creating
 *       reusable class that will be used in both places.
 */
class TransposeTensorAdapterBase {
public:
    TransposeTensorAdapterBase(cudaDataType_t element_tpe,
                               size_t element_size,
                               const std::vector<int64_t>& src_shape,
                               const std::vector<int64_t>& dst_shape,
                               const std::vector<int>& mode);

    void requestWorkbuffer(std::vector<size_t>& workbuffers_sizes);

    void* dnnApiPtr(const std::vector<Workbuffers::mutable_buffer>& mutable_buffers) const;

protected:
    void execute(const InferenceRequestContext& context, const void* src, void* dst) const;

    WorkbufferDesc workbuffer_;

private:
    void initCuTensorDescriptor(const CUDA::CuTensorHandle& handle,
                                const std::vector<int64_t>& shape,
                                cutensorTensorDescriptor_t& desc) const;

    cudaDataType_t element_type_;
    size_t element_size_;
    std::vector<int64_t> src_shape_;
    std::vector<int64_t> dst_shape_;
    std::vector<int> src_mode_;
    std::vector<int> dst_mode_;
};

class TransposeInputTensorAdapter : public TransposeTensorAdapterBase {
public:
    using TransposeTensorAdapterBase::TransposeTensorAdapterBase;
    void execute(const InferenceRequestContext& context,
                 CUDA::DevicePointer<const void*> input,
                 const std::vector<Workbuffers::mutable_buffer>& dst) const;
};

class TransposeOutputTensorAdapter : public TransposeTensorAdapterBase {
public:
    using TransposeTensorAdapterBase::TransposeTensorAdapterBase;
    void execute(const InferenceRequestContext& context,
                 const std::vector<Workbuffers::mutable_buffer>& src,
                 CUDA::DevicePointer<void*> output) const;
};

}  // namespace CUDAPlugin::RNN::Details
