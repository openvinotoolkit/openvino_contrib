// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_color_i420.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(I420toRGBOp, I420toRGB);
OPERATION_REGISTER(I420toBGROp, I420toBGR);

}  // namespace nvidia_gpu
}  // namespace ov
