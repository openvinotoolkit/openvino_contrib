// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prelu.hpp"

#include <fmt/format.h>

#include <gsl/gsl_assert>

#include "cuda_operation_registry.hpp"

namespace CUDAPlugin {

OPERATION_REGISTER(PReluOp, PRelu)

}  // namespace CUDAPlugin
