// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_from_mlir(mlir::ModuleOp module, const BaseCodegenDesc& desc) {
    switch (desc.kind) {
        case KernelOpKind::MatMul:
            return generate_msl_for_matmul(static_cast<const MatMulCodegenDesc&>(desc), module);
        case KernelOpKind::Conv2D:
            return generate_msl_for_conv2d(static_cast<const Conv2DCodegenDesc&>(desc), module);
        case KernelOpKind::Conv3D:
            return generate_msl_for_conv3d(static_cast<const Conv3DCodegenDesc&>(desc), module);
        case KernelOpKind::ElementwiseAdd:
        case KernelOpKind::ElementwiseSub:
        case KernelOpKind::ElementwiseMul:
        case KernelOpKind::ElementwiseDiv:
        case KernelOpKind::ElementwisePow:
        case KernelOpKind::ElementwiseMod:
        case KernelOpKind::ElementwiseFloorMod:
            return generate_msl_for_eltwise(static_cast<const EltwiseCodegenDesc&>(desc), module);
        case KernelOpKind::MaxPool2D:
            return generate_msl_for_maxpool2d(static_cast<const Pool2DCodegenDesc&>(desc), module);
        case KernelOpKind::AvgPool2D:
            return generate_msl_for_avgpool2d(static_cast<const Pool2DCodegenDesc&>(desc), module);
        case KernelOpKind::Softmax:
            return generate_msl_for_softmax(static_cast<const SoftmaxCodegenDesc&>(desc), module);
        default:
            OPENVINO_THROW("MLIR->MSL codegen: unsupported op kind");
    }
}

}  // namespace metal_plugin
}  // namespace ov
