// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>

#include <gpu/gpu_context_api_cuda.hpp>
#include <gpu/device_pointers.hpp>
#include <ie_layouts.h>

namespace ngraph {

class Node;

}

namespace CUDAPlugin {

using InferenceEngine::gpu::InferenceRequestContext;

class IOperationExec {
 public:
  using Inputs = gsl::span<InferenceEngine::gpu::DevicePointer<const void*>>;
  using Outputs = gsl::span<InferenceEngine::gpu::DevicePointer<void*>>;

  virtual ~IOperationExec() = default;
  virtual void Execute(const InferenceRequestContext& context,
                       Inputs inputTensors,
                       Outputs outputTensors) = 0;
};

class IOperationMeta {
 public:
  virtual ~IOperationMeta() = default;
  virtual std::string GetName() const = 0;
  virtual gsl::span<const unsigned> GetInputIds() const = 0;
  virtual gsl::span<const unsigned> GetOutputIds() const = 0;
};

class OperationBase
    : public IOperationExec
    , public IOperationMeta
    , public std::enable_shared_from_this<OperationBase> {
 public:
  using Ptr = std::shared_ptr<OperationBase>;
  using WeakPtr = std::weak_ptr<OperationBase>;

  OperationBase(const std::shared_ptr<ngraph::Node>& node,
                std::vector<unsigned> inputIds,
                std::vector<unsigned> outputIds);

  std::string GetName() const override;
  gsl::span<const unsigned> GetInputIds() const override;
  gsl::span<const unsigned> GetOutputIds() const override;

 protected:
  std::string node_name_;
  const std::vector<unsigned> input_ids_;
  const std::vector<unsigned> output_ids_;
};

inline
std::string OperationBase::GetName() const {
  return node_name_;
}

inline
gsl::span<const unsigned>
OperationBase::GetInputIds() const {
  return input_ids_;
}

inline
gsl::span<const unsigned>
OperationBase::GetOutputIds() const {
  return output_ids_;
}

} // namespace CUDAPlugin
