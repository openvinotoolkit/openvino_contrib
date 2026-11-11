// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// PointPillars OpenVINO Extension library entry point
// Registers custom operations for PointPillars

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

#include "pp_postprocess.hpp"
#include "pp_scatter.hpp"
#include "pp_voxelization.hpp"

OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        std::make_shared<ov::OpExtension<PpExtension::PPVoxelizationScatter>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPVoxelizationFinalize>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPVoxelizationCoords>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPVoxelizationParams>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPVoxelizationCellPillar>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPScatterBEV>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPPostprocScores>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPPostprocCandidates>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPPostprocNms>>(),
        std::make_shared<ov::OpExtension<PpExtension::PPPostprocGather>>()
    }));
