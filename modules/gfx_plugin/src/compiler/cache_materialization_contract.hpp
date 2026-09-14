// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "compiler/cache_envelope.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

CacheMaterializationContract make_cache_materialization_contract(
    const RuntimeExecutableDescriptor &descriptor);

std::string serialize_cache_materialization_contract(
    const CacheMaterializationContract &contract);

CacheMaterializationContract deserialize_cache_materialization_contract(
    std::string_view wire, std::vector<std::string> &diagnostics);

RuntimeExecutableDescriptor apply_cache_materialization_contract(
    const CacheMaterializationContract &contract,
    RuntimeExecutableDescriptor descriptor);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
