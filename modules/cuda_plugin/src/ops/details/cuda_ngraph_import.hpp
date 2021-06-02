// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef __NVCC__
/*
 * This workaround is used to avoid issues with CUDA Compiler.
 * Current version of CUDA Compiler does not understand __attribute__((deprecated((msg)))).
 * To avoid such issue we temporary define deprecated macro definition and after include undefine it again
 */
#pragma push_macro("deprecated")
#undef deprecated
#define deprecated(...)

#include <ngraph/node.hpp>
#include <ngraph/op/split.hpp>
#include <ngraph/op/constant.hpp>

#pragma pop_macro("deprecated")

#else

#error "Do not use this header file in *.cpp files !! Intent of this file is to be used inside of *.cu files"

#endif
