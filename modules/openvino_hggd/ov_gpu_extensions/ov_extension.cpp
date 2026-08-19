/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */
// OpenVINO Extension Entry Point for HGGD Point Cloud Operations

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "pointcloud_ops.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // ═══ GPU-Compatible Single-Output Operations ═══
        // These pack multi-output results into single tensor for GPU custom layer support
        
        // KNNPointsSingle - [B, N1, K*2] output (dists + indices)
        std::make_shared<ov::OpExtension<HGGDExtension::KNNPointsSingle>>(),
        std::make_shared<ov::frontend::OpExtension<HGGDExtension::KNNPointsSingle>>(),
        
        // BallQuerySingle - [B, N1, K*2] output (dists + indices)
        std::make_shared<ov::OpExtension<HGGDExtension::BallQuerySingle>>(),
        std::make_shared<ov::frontend::OpExtension<HGGDExtension::BallQuerySingle>>(),
        
        // FPSSingle - [B, K, 4] output (xyz + index)
        std::make_shared<ov::OpExtension<HGGDExtension::FPSSingle>>(),
        std::make_shared<ov::frontend::OpExtension<HGGDExtension::FPSSingle>>(),
        
        // FPSWithLengths - [B, K, 4] output with per-batch lengths support
        std::make_shared<ov::OpExtension<HGGDExtension::FPSWithLengths>>(),
        std::make_shared<ov::frontend::OpExtension<HGGDExtension::FPSWithLengths>>(),
    }));
