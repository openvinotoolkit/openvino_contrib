// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <unordered_map>

#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

class OperationRegistry final {
 public:
    template<typename TOperation>
    class Register {
     public:
        static_assert(std::is_base_of<OperationBase, TOperation>::value,
                      "TOperation should derive from OperationBase");

        Register(const std::string& opName, const std::string& typeName);
    };

    static OperationRegistry& getInstance();

    bool hasOperation(const std::shared_ptr<ngraph::Node>& node);
    OperationBase::Ptr createOperation(const std::shared_ptr<ngraph::Node>& node,
                                       const std::vector<unsigned>& inIds,
                                       const std::vector<unsigned>& outIds);
    const std::string& getOperationTypename(const std::shared_ptr<ngraph::Node>& node);

 private:
    using OperationBuilder = std::function<OperationBase::Ptr(const std::shared_ptr<ngraph::Node>&,
                                                              std::vector<unsigned>,
                                                              std::vector<unsigned>)>;
    bool hasOperation(const std::string &name);

    std::unordered_map<std::string, OperationBuilder> registered_operations_;
    std::unordered_map<std::string, std::string> operations_type_name_;
};

template<typename TOperation>
OperationRegistry::Register<TOperation>::Register(const std::string& opName, const std::string& typeName) {
  auto& opRegistry = OperationRegistry::getInstance();
  if (opRegistry.hasOperation(opName)) {
    throw std::runtime_error{"Operation " + opName + " is already registered !!"};
  }
  opRegistry.registered_operations_.insert({
    opName,
    [](const std::shared_ptr<ngraph::Node>& node,
       const std::vector<unsigned>& inputs,
       const std::vector<unsigned>& outputs) {
      return std::make_shared<TOperation>(node, inputs, outputs);
    }});
  opRegistry.operations_type_name_.insert({opName, typeName});
}

#define OPERATION_REGISTER(type, name) \
  [[maybe_unused]] auto reg##type = OperationRegistry::Register<type>(name, #type);

} // namespace CUDAPlugin
