// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/backend_registry.hpp"

#include <utility>

namespace ov {
namespace gfx_plugin {
namespace compiler {

BackendRegistry::BackendRegistry(
    std::vector<std::shared_ptr<const BackendModule>> modules)
    : m_modules(std::move(modules)) {}

std::shared_ptr<const BackendModule>
BackendRegistry::resolve(const BackendTarget &target) const {
  const auto fingerprint = target.fingerprint();
  for (const auto &module : m_modules) {
    if (module &&
        module->target().is_compatible_with_fingerprint(fingerprint)) {
      return module;
    }
  }
  return {};
}

std::vector<BackendTarget> BackendRegistry::available_targets() const {
  std::vector<BackendTarget> targets;
  targets.reserve(m_modules.size());
  for (const auto &module : m_modules) {
    if (module) {
      targets.push_back(module->target());
    }
  }
  return targets;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
