// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"

namespace ov {
namespace nvidia_gpu {

OperationRegistry& OperationRegistry::getInstance() {
    static OperationRegistry registry;
    return registry;
}

void OperationRegistry::registerOp(const std::string& opName, OperationBuilder&& builder) {
    if (!registered_operations_.try_emplace(opName, move(builder)).second)
        throw std::runtime_error{"Operation " + opName + " is already registered !!"};
}

bool OperationRegistry::hasOperation(const std::shared_ptr<ov::Node>& node) {
    return hasOperation(node->get_type_info().name);
}

std::optional<std::type_index> OperationRegistry::getOperationType(const std::shared_ptr<ov::Node>& node) const {
    if (registered_type_operations_.count(node->get_type_info().name) > 0) {
        return registered_type_operations_.at(node->get_type_info().name);
    }
    return std::nullopt;
}

bool OperationRegistry::hasOperation(const std::string& name) {
    return registered_operations_.end() != registered_operations_.find(name);
}

OperationBase::Ptr OperationRegistry::createOperation(const CreationContext& context,
                                                      const std::shared_ptr<ov::Node>& node,
                                                      std::vector<TensorID>&& inIds,
                                                      std::vector<TensorID>&& outIds) {
    auto& opBuilder = registered_operations_.at(node->get_type_info().name);
    return opBuilder(context, node, move(inIds), move(outIds));
}

OperationBase::Ptr OperationRegistry::createOperation(const CreationContext& context,
                                                      const std::shared_ptr<ov::Node>& node,
                                                      gsl::span<const TensorID> inIds,
                                                      gsl::span<const TensorID> outIds) {
    auto toVector = [](gsl::span<const TensorID> s) { return std::vector<TensorID>(s.begin(), s.end()); };
    return createOperation(context, node, toVector(inIds), toVector(outIds));
}

}  // namespace nvidia_gpu
}  // namespace ov
