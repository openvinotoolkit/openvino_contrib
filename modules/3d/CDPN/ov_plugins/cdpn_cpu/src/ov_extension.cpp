// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Extension entry point - registers all CDPN custom ops with OV runtime.

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>

#include "cdpn_pnp_solve.hpp"
#include "cdpn_preprocess.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::OpExtension<CdpnExtension::CdpnPreprocess>>(),
    std::make_shared<ov::OpExtension<CdpnExtension::CdpnPnpSolve>>(),
}));
