// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "compiler/backend_registry.hpp"
#include "compiler/backend_target.hpp"
#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {

struct RuntimeExecutableDescriptor;

namespace compiler {

struct CacheImportContract {
  BackendTarget target;
  ExecutableBundle executable;
  std::shared_ptr<const RuntimeExecutableDescriptor> runtime_descriptor;
  std::shared_ptr<const ov::Model> runtime_model;
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

CacheImportContract
make_cache_import_contract(const CacheEnvelope &envelope,
                           const BackendRegistry &registry);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
