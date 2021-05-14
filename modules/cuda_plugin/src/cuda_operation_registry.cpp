// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"

#include <ngraph/node.hpp>

namespace CUDAPlugin {

OperationRegistry& OperationRegistry::getInstance() {
  static OperationRegistry registry;
  return registry;
}

void OperationRegistry::registerOp(const std::string& opName,
                                   OperationBuilder&& builder) {
  if (hasOperation(opName)) {
    throw std::runtime_error{"Operation " + opName +
                             " is already registered !!"};
  }
  registered_operations_.emplace(opName, move(builder));
}

bool OperationRegistry::hasOperation(
    const std::shared_ptr<ngraph::Node>& node) {
  return hasOperation(node->get_type_info().name);
}

bool OperationRegistry::hasOperation(const std::string& name) {
  return registered_operations_.end() != registered_operations_.find(name);
}

OperationBase::Ptr OperationRegistry::createOperation(
    const std::shared_ptr<ngraph::Node>& node, std::vector<unsigned>&& inIds,
    std::vector<unsigned>&& outIds) {
  auto& opBuilder = registered_operations_.at(node->get_type_info().name);
  return opBuilder(node, move(inIds), move(outIds));
}

OperationBase::Ptr OperationRegistry::createOperation(
    const std::shared_ptr<ngraph::Node>& node, gsl::span<const unsigned> inIds,
    gsl::span<const unsigned> outIds) {
  auto toVector = [](gsl::span<const unsigned> s) {
    return std::vector<unsigned>(s.begin(), s.end());
  };
  return createOperation(node, toVector(inIds), toVector(outIds));
}

}  // namespace CUDAPlugin
