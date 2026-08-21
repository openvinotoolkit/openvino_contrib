// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "furthest_point_sampling.hpp"
#include "gather_operation.hpp"
#include "ball_query.hpp"
#include "grouping_operation.hpp"
#include "custom_svd_u.hpp"
#include "custom_svd_v.hpp"
#include "custom_det.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::FurthestPointSampling>>(),
        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::FurthestPointSampling>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GatherOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GatherOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::GroupingOperation>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::GroupingOperation>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::BallQuery>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::BallQuery>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVDu>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVDu>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomSVDv>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomSVDv>>(),

        std::make_shared<ov::OpExtension<TemplateExtension::CustomDet>>(),
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::CustomDet>>(),

    }));
//! [ov_extension:entry_point]
// clang-format on
