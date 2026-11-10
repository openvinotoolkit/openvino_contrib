/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// OpenVINO Extension Entry Point for SparseConv3d Engine

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "sparse_conv_3d_op.hpp"
#include "flexicubes_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<SAM3DExtension::SparseConv3dOp>>(),
        std::make_shared<ov::frontend::OpExtension<SAM3DExtension::SparseConv3dOp>>(),
        std::make_shared<ov::OpExtension<SAM3DExtension::SparseMeshUpsampleOp>>(),
        std::make_shared<ov::frontend::OpExtension<SAM3DExtension::SparseMeshUpsampleOp>>(),
        std::make_shared<ov::OpExtension<SAM3DExtension::FlexiCubesOp>>(),
        std::make_shared<ov::frontend::OpExtension<SAM3DExtension::FlexiCubesOp>>(),
    }));
