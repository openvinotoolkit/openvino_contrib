// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/interpolate.hpp"

namespace ov::nvidia_gpu::Interpolate::Details {

void getAxesAndScales(const ov::op::v4::Interpolate& node, std::vector<size_t>& axes, std::vector<float>& scales);

}  // namespace ov::nvidia_gpu::Interpolate::Details
