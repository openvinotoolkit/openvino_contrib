// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_operation_registry.hpp>
#include "convert_color_i420.hpp"

namespace CUDAPlugin {

OPERATION_REGISTER(I420toRGBOp, I420toRGB);
OPERATION_REGISTER(I420toBGROp, I420toBGR);

}
