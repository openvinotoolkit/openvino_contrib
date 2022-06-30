// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_operation_registry.hpp>
#include "convert_color_nv12.hpp"

namespace CUDAPlugin {

OPERATION_REGISTER(NV12toRGBOp, NV12toRGB);
OPERATION_REGISTER(NV12toBGROp, NV12toBGR);

}
