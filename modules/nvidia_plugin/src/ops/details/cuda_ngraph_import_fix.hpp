// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/deprecated.hpp>

#undef NGRAPH_DEPRECATED
#define NGRAPH_DEPRECATED(msg) __attribute__((deprecated(msg)))
