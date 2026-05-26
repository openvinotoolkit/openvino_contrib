// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/kernel_registry.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

namespace {

KernelUnit make_opencl_generated_kernel_unit(const BackendTarget& target) {
    return KernelUnit::describe(LoweringRouteKind::GeneratedKernel,
                                KernelUnitKind::GeneratedKernel,
                                "opencl_generated_kernel",
                                target.backend_id(),
                                "opencl_generated_kernel");
}

KernelUnit make_opencl_exception_unit(const BackendTarget& target) {
    return KernelUnit::describe(LoweringRouteKind::HandwrittenKernelException,
                                KernelUnitKind::HandwrittenException,
                                "opencl_handwritten_exception",
                                target.backend_id(),
                                "opencl_source_exception");
}

}  // namespace

KernelRegistry make_opencl_kernel_registry(const BackendTarget& target) {
    auto units = make_common_kernel_units(target);
    units.push_back(make_opencl_generated_kernel_unit(target));
    units.push_back(make_opencl_exception_unit(target));
    return KernelRegistry(target, std::move(units));
}

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
