// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_operation_support.hpp"

#include <utility>

#include "backends/opencl/compiler/opencl_kernel_unit_catalog.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

OperationSupportResult
query_opencl_operation(const std::shared_ptr<const ov::Node> &node,
                       const KernelRegistry &kernel_registry) {
  for (const auto &entry : opencl_operation_support_entries()) {
    if (!entry.matches(node)) {
      continue;
    }
    if (entry.query_support) {
      return entry.query_support(node, kernel_registry);
    }
    return make_unsupported_operation(entry.unsupported_reason);
  }
  return make_unsupported_operation("missing_opencl_kernel_unit");
}

class OpenCLOperationSupportPolicy final : public OperationSupportPolicy {
public:
  explicit OpenCLOperationSupportPolicy(KernelRegistry kernel_registry)
      : m_kernel_registry(std::move(kernel_registry)) {}

  OperationSupportResult
  query_operation(const OperationSupportQuery &query) const override {
    return query_opencl_operation(query.node, m_kernel_registry);
  }

private:
  KernelRegistry m_kernel_registry;
};

} // namespace

std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy(KernelRegistry kernel_registry) {
  return std::make_shared<OpenCLOperationSupportPolicy>(
      std::move(kernel_registry));
}

std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy() {
  return make_opencl_operation_support_policy(make_opencl_kernel_registry(
      BackendTarget::from_backend(GpuBackend::OpenCL)));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
