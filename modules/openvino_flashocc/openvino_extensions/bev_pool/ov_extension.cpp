/*
 * Copyright (C) 2018-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 * Adapted from OpenVINO BEVFusion module BEVPool sources
 * (https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/openvino_bevfusion), Apache-2.0.
 * BEV pooling algorithm adapted from BEVDet bev_pool_v2
 * (https://github.com/HuangJunJie2017/BEVDet), Apache-2.0, Copyright (c) Phigent Robotics.
 */

// OpenVINO Extension Entry Point for BEVPool Operations

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/onnx/extension/op.hpp>

#include "bev_pool_op.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Original fused BEVPool (float CAS atomics)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPool>>(),
        std::make_shared<ov::frontend::onnx::OpExtension<BEVFusionExtension::BEVPool>>(
            "BEVPool", "bevfusion", std::map<std::string, std::string>{}, std::map<std::string, ov::Any>{
                {"nx", int64_t(200)}, {"ny", int64_t(200)}, {"nz", int64_t(1)},
                {"x_min", -40.0f}, {"y_min", -40.0f}, {"z_min", -1.0f},
                {"x_step", 0.4f}, {"y_step", 0.4f}, {"z_step", 6.4f},
                {"channels", int64_t(64)}
            }),
        // Optimized: BEVPoolScatter (int32 native atomics)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPoolScatter>>(),
        std::make_shared<ov::frontend::onnx::OpExtension<BEVFusionExtension::BEVPoolScatter>>(
            "BEVPoolScatter", "bevfusion", std::map<std::string, std::string>{}, std::map<std::string, ov::Any>{
                {"nx", int64_t(200)}, {"ny", int64_t(200)}, {"nz", int64_t(1)},
                {"x_min", -40.0f}, {"y_min", -40.0f}, {"z_min", -1.0f},
                {"x_step", 0.4f}, {"y_step", 0.4f}, {"z_step", 6.4f},
                {"channels", int64_t(64)}, {"scale", int64_t(8192)}
            }),
        // Optimized: BEVPoolConvert (int32 → float32)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPoolConvert>>(),
        std::make_shared<ov::frontend::onnx::OpExtension<BEVFusionExtension::BEVPoolConvert>>(
            "BEVPoolConvert", "bevfusion", std::map<std::string, std::string>{},
            std::map<std::string, ov::Any>{{"scale", int64_t(8192)}}),
        // V2: pre-sorted interval-based scatter (no atomics)
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPoolV2>>(),
        std::make_shared<ov::frontend::onnx::OpExtension<BEVFusionExtension::BEVPoolV2>>(
            "BEVPoolV2", "bevfusion", std::map<std::string, std::string>{
                {"nx", "nx"}, {"ny", "ny"}, {"channels", "channels"},
                {"feat_hw", "feat_hw"}, {"depth_hw", "depth_hw"}, {"total_pts", "total_pts"}
            }),
        // V2 GPU sorting: counting sort of geometry
        std::make_shared<ov::OpExtension<BEVFusionExtension::BEVPoolBinSort>>(),
        std::make_shared<ov::frontend::onnx::OpExtension<BEVFusionExtension::BEVPoolBinSort>>(
            "BEVPoolBinSort", "bevfusion", std::map<std::string, std::string>{
                {"nx", "nx"}, {"ny", "ny"}, {"total_pts", "total_pts"},
                {"x_min", "x_min"}, {"y_min", "y_min"},
                {"x_step", "x_step"}, {"y_step", "y_step"}
            }),
    }));
