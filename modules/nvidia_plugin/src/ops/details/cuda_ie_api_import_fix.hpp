// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#undef INFERENCE_ENGINE_DEPRECATED
#define INFERENCE_ENGINE_DEPRECATED(msg) __attribute__((deprecated(msg)))
