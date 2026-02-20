// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/parameter.hpp>
#include <typeinfo>

struct ParameterStubNode : ov::op::v0::Parameter {
    using ov::op::v0::Parameter::Parameter;
    OPENVINO_OP("Parameter");

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<ParameterStubNode>();
    }
};
