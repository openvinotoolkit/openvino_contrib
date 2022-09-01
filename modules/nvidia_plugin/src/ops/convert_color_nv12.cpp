// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_nv12.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(NV12toRGBOp, NV12toRGB);
OPERATION_REGISTER(NV12toBGROp, NV12toBGR);

}  // namespace nvidia_gpu
}  // namespace ov
