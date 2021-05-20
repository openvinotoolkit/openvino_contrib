// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <gsl/span>
#include <unordered_map>

#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

class OperationRegistry final {
 public:
  using OperationBuilder = std::function<OperationBase::Ptr(
      const std::shared_ptr<ngraph::Node>&, std::vector<unsigned>&&,
      std::vector<unsigned>&&)>;
  template <typename TOperation>
  class Register {
   public:
    static_assert(std::is_base_of_v<OperationBase, TOperation>,
                  "TOperation should derive from OperationBase");

    explicit Register(const std::string& opName) {
      getInstance().registerOp(
          opName,
          [](const std::shared_ptr<ngraph::Node>& node,
             std::vector<unsigned>&& inputs, std::vector<unsigned>&& outputs) {
            return std::make_shared<TOperation>(node, move(inputs),
                                                move(outputs));
          });
    }
  };

  static OperationRegistry& getInstance();

  bool hasOperation(const std::shared_ptr<ngraph::Node>& node);

  OperationBase::Ptr createOperation(const std::shared_ptr<ngraph::Node>& node,
                                     std::vector<unsigned>&& inIds,
                                     std::vector<unsigned>&& outIds);

  OperationBase::Ptr createOperation(const std::shared_ptr<ngraph::Node>& node,
                                     gsl::span<const unsigned> inIds,
                                     gsl::span<const unsigned> outIds);

 private:
  void registerOp(const std::string& opName, OperationBuilder&& builder);

  bool hasOperation(const std::string& name);

  std::unordered_map<std::string, OperationBuilder> registered_operations_;
};

}  // namespace CUDAPlugin

#define OPERATION_REGISTER(type, name)                                    \
  [[maybe_unused]] ::CUDAPlugin::OperationRegistry::Register<type> \
      op_register_##name{#name};
