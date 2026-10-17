// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include "backends/opencl/compiler/opencl_kernel_unit_catalog.hpp"

#include <string>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_opencl_generated_kernel_unit(const BackendTarget &target,
                                             std::string unit_id,
                                             std::string op_family) {
  return KernelUnit::describe(LoweringRouteKind::GeneratedKernel,
                              KernelUnitKind::GeneratedKernel,
                              std::move(unit_id),
                              target.backend_id(), std::move(op_family));
}

} // namespace

KernelRegistry make_opencl_kernel_registry(const BackendTarget &target) {
  auto units = make_common_kernel_units(target);
  for (const auto &spec : opencl_generated_kernel_unit_specs()) {
    units.push_back(make_opencl_generated_kernel_unit(
        target, std::string(spec.kernel_unit_id), std::string(spec.op_family)));
  }
  return KernelRegistry(target, std::move(units));
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
