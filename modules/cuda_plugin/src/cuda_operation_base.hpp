// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>

#include <cuda_creation_context.hpp>
#include <cuda/device_pointers.hpp>
#include <cuda_inference_request_context.hpp>
#include <memory>
#include <memory_manager/model/cuda_memory_model.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "memory_manager/cuda_workbuffers.hpp"

namespace ngraph {

class Node;

}

namespace CUDAPlugin {

class IOperationExec {
public:
    using Inputs = gsl::span<const CUDA::DevicePointer<const void*>>;
    using Outputs = gsl::span<const CUDA::DevicePointer<void*>>;
    using Buffers = std::vector<CUDA::DevicePointer<void*>>;
    enum class WorkbufferStatus { NoInitNeeded, InitNeeded };

    virtual ~IOperationExec() = default;
    virtual void Execute(const InferenceRequestContext& context,
                         Inputs inputTensors,
                         Outputs outputTensors,
                         const Workbuffers& workbuffers) const = 0;
    virtual void InitSharedImmutableWorkbuffers(const Buffers&) = 0;
    virtual WorkbufferRequest GetWorkBufferRequest() const = 0;
    virtual const WorkbufferIds& GetWorkbufferIds() const = 0;
    virtual WorkbufferStatus SetWorkbufferIds(WorkbufferIds&& workbufferIds) = 0;
};

class IOperationMeta {
public:
    struct Category {
        static constexpr std::string_view CUDA{"CUDA"};
        static constexpr std::string_view cuBLAS{"cuBLAS"};
        static constexpr std::string_view cuDNN{"cuDNN"};
        static constexpr std::string_view cuTENSOR{"cuTENSOR"};
    };

    virtual ~IOperationMeta() = default;
    virtual const std::string_view& GetCategory() const = 0;
    virtual const std::string& GetName() const = 0;
    virtual const std::string& GetTypeName() const = 0;
    virtual gsl::span<const TensorID> GetInputIds() const = 0;
    virtual gsl::span<const TensorID> GetOutputIds() const = 0;
};

class OperationBase : public IOperationExec, public IOperationMeta, public std::enable_shared_from_this<OperationBase> {
public:
    using Ptr = std::shared_ptr<OperationBase>;
    using WeakPtr = std::weak_ptr<OperationBase>;
    using IndexCollection = std::vector<TensorID>;
    OperationBase(const CreationContext& context,
                  const ngraph::Node& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);

    WorkbufferRequest GetWorkBufferRequest() const override {
        return {};  // Most operators do not need workbuffers
    }
    void InitSharedImmutableWorkbuffers(const Buffers&) override {}

protected:
    OperationBase(const CreationContext& context,
                  const std::shared_ptr<ngraph::Node>& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds)
        : OperationBase(context, *node, move(inputIds), move(outputIds)) {}

public:
    const std::string_view& GetCategory() const override { return Category::CUDA; }
    const std::string& GetName() const override { return node_name_; }
    const std::string& GetTypeName() const override { return type_name_; }
    gsl::span<const TensorID> GetInputIds() const override { return input_ids_; }
    gsl::span<const TensorID> GetOutputIds() const override { return output_ids_; }
    const WorkbufferIds& GetWorkbufferIds() const override { return workbuffer_ids_; }
    WorkbufferStatus SetWorkbufferIds(WorkbufferIds&& workbufferIds) override {
        workbuffer_ids_ = workbufferIds;
        return workbuffer_ids_.immutableIds.empty() ? WorkbufferStatus::NoInitNeeded : WorkbufferStatus::InitNeeded;
    }

protected:
    std::string node_name_;
    std::string type_name_;
    const IndexCollection input_ids_;
    const IndexCollection output_ids_;
    WorkbufferIds workbuffer_ids_;
};

template <decltype(&IOperationMeta::Category::CUDA) CategoryString>
class CategorizedOperationBase : public OperationBase {
protected:
    using OperationBase::OperationBase;

public:
    const std::string_view& GetCategory() const override { return *CategoryString; }
};

using OperationCuDnn = CategorizedOperationBase<&IOperationMeta::Category::cuDNN>;
using OperationCuBlas = CategorizedOperationBase<&IOperationMeta::Category::cuBLAS>;
using OperationCuTensor = CategorizedOperationBase<&IOperationMeta::Category::cuTENSOR>;

/**
 * @brief Downcasts a shared node pointer to a ConcreteOperator reference
 */
template <class ConcreteOperator>
ConcreteOperator& downcast(const std::shared_ptr<ngraph::Node>& node) {
    return dynamic_cast<ConcreteOperator&>(*node.get());
}
}  // namespace CUDAPlugin
