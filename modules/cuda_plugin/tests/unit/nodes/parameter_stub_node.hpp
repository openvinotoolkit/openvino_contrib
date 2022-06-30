// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeinfo>
#include <ngraph/node.hpp>
#include <openvino/op/parameter.hpp>

struct ParameterStubNode : ov::op::v0::Parameter {
    using ov::op::v0::Parameter::Parameter;
    inline static constexpr type_info_t type_info{"Parameter", 0ul};
    const type_info_t& get_type_info() const override {
        return type_info;
    }

    std::shared_ptr<ov::Node>
    clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<ParameterStubNode>();
    }
};