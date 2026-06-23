// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "compiler/kernel_registry.hpp"
#include "compiler/operation_support.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy(KernelRegistry kernel_registry);
std::shared_ptr<const OperationSupportPolicy>
make_opencl_operation_support_policy();

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
