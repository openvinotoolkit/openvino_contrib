/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */


// OpenVINO Extension Entry Point for Vox2Seq

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "vox2seq_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<SAM3DExtension::Vox2SeqOp>>(),
        std::make_shared<ov::frontend::OpExtension<SAM3DExtension::Vox2SeqOp>>(),
    }));
