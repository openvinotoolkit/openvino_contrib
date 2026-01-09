// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

namespace ov {
namespace nvidia_gpu {

/**
 * @brief NOP - no operation. Common implementation for all operations which
 * do nothing.
 *
 * These operations are at least the following: Reshape, Squeeze, Unsqueeze,
 * Constant.
 *
 * The purpose of having NOP operations in execution queue is to make them
 * transparent for the rest of plugin implementation, so they don't require
 * special handling to skip their execution.
 *
 * Note, that Reshape-like operations do not need to perform any data copying
 * because their input and output data tensors reuse the same memory allocation.
 * Constants also have nothing to do, because at the time of execution their
 * values are already copied to device side and linked with all dependent
 * consumer operations.
 */
class NopOp : public OperationBase {
public:
    using OperationBase::OperationBase;

    gsl::span<const TensorID> GetInputIds() const override { return gsl::span<const TensorID>{}; };

    gsl::span<const TensorID> GetOutputIds() const override { return gsl::span<const TensorID>{}; };

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override {}

    CudaGraphCompatibility GetCudaGraphCompatibility() const override { return CudaGraphCompatibility::FULL; }
};

}  // namespace nvidia_gpu
}  // namespace ov
