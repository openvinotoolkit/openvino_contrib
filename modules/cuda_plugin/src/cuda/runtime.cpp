// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime.hpp"

#include <details/ie_exception.hpp>

namespace CUDA {
[[gnu::cold, noreturn]] void throwIEException(
    const std::string& msg,
    const std::experimental::source_location& location) {
  throw InferenceEngine::details::InferenceEngineException(
      location.file_name(), location.line(), msg);
}

[[gnu::cold]] void logError(
    const std::string& /*msg*/,
    const std::experimental::source_location& /*location*/) {
}  // TODO: log somewhere

}  // namespace CUDA
