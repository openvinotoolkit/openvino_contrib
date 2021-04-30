// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/node.hpp>

#include "cuda_operation_registry.hpp"

namespace CUDAPlugin {

OperationRegistry&
OperationRegistry::getInstance() {
    static OperationRegistry registry{};
    return registry;
}

bool OperationRegistry::hasOperation(const std::shared_ptr<ngraph::Node>& node) {
    return hasOperation(node->get_type_info().name);
}

bool OperationRegistry::hasOperation(const std::string& name) {
    return registered_operations_.end() !=
           registered_operations_.find(name);
}

OperationBase::Ptr
OperationRegistry::createOperation(const std::shared_ptr<ngraph::Node>& node,
                                   const std::vector<unsigned>& inIds,
                                   const std::vector<unsigned>& outIds) {
    auto& opBuilder = registered_operations_.at(node->get_type_info().name);
    return opBuilder(node, inIds, inIds);
}

const std::string&
OperationRegistry::getOperationTypename(const std::shared_ptr<ngraph::Node>& node) {
    return operations_type_name_.at(node->get_type_info().name);
}

} // namespace CUDAPlugin
