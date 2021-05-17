// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header that defines wrappers for internal GPU plugin-specific
 * CUDA context and CUDA shared memory blobs
 *
 * @file gpu_context_api_cuda.hpp
 */
#pragma once

#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <ie_remote_context.hpp>
#include <cuda/stream.hpp>
#include <utility>

#include "gpu/gpu_params.hpp"
#include "gpu/details/gpu_context_helpers.hpp"

namespace InferenceEngine {

namespace gpu {
/**
 * @brief This class represents an abstraction for GPU plugin remote context
 * which is shared with OpenCL context object.
 * The plugin object derived from this class can be obtained either with
 * GetContext() method of Executable network or using CreateContext() Core call.
 */
class CudaContext : public RemoteContext, public details::param_map_obj_getter {
 public:
  /**
   * @brief A smart pointer to the ClContext object
   */
  using Ptr = std::shared_ptr<CudaContext>;
  using WeakPtr = std::weak_ptr<CudaContext>;

  // TODO: Add additional functions
};

class InferenceRequestContext {
 public:
  /**
   * @brief A smart pointer to the InferenceRequestContext object
   */
  using Ptr = std::shared_ptr<InferenceRequestContext>;
  using WeakPtr = std::weak_ptr<InferenceRequestContext>;

  InferenceRequestContext(int deviceId,
                          std::shared_ptr<CUDAPlugin::CudaStream> stream,
                          const InferenceEngine::BlobMap& inputs,
                          const InferenceEngine::BlobMap& outputs)
  : device_id_{deviceId}
  , cuda_stream {stream}
  , blob_inputs {inputs}
  , blob_outputs {outputs} {}

  /**
   * @brief GetInputBlob(name) returns an input blob with the given name
   */
  Blob::Ptr GetInputBlob(const std::string& input_name) const {
    return blob_inputs.at(input_name);
  }
  /**
   * @brief GetInputBlob(name) returns an input blob with the given name
   */
  Blob::Ptr GetOutputBlob(const std::string& input_name) const {
    return blob_outputs.at(input_name);
  }
  /**
   * @brief HasInputBlob(name) returns true if it contains an input blob with the given name
   */
  bool HasInputBlob(const std::string& input_name) const noexcept {
    return blob_inputs.find(input_name) != blob_inputs.end();
  }
  /**
   * @brief HasOutputBlob(name) returns true if contains an output blob with the given name
   */
  bool HasOutputBlob(const std::string& input_name) const noexcept {
    return blob_outputs.find(input_name) != blob_outputs.end();
  }
  /**
   * Returns CUDA Device Id that is mapped to appropriate
   * CUDADevice index returned by CudaDevice::GetAllDevicesProp()
   * function
   * @return Cuda Device Index
   */
  int GetDeviceId() const noexcept {
    return device_id_;
  }
  /**
   * @brief GetCUDAStream() returns associated CUDA stream
   */
  std::shared_ptr<CUDAPlugin::CudaStream>
  GetCUDAStream() const noexcept {
    return cuda_stream;
  }
 private:
    int device_id_;
    std::shared_ptr<CUDAPlugin::CudaStream> cuda_stream;
    const InferenceEngine::BlobMap& blob_inputs;
    const InferenceEngine::BlobMap& blob_outputs;
};

} // namespace gpu

} // namespace InferenceEngine
