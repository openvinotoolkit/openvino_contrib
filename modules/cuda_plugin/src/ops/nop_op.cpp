// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nop_op.hpp"
#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

OperationRegistry::Register<NopOp> op_registrar_Constant{"Constant"};
OperationRegistry::Register<NopOp> op_registrar_Reshape{"Reshape"};
OperationRegistry::Register<NopOp> op_registrar_Squeeze{"Squeeze"};
OperationRegistry::Register<NopOp> op_registrar_Unsqueeze{"Unsqueeze"};

} // namespace CUDAPlugin
