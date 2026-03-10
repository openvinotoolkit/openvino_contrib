// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
inline std::ostream& operator<<(std::ostream& os, nullptr_t) {  // insert a null pointer
    return os << "nullptr";
}
