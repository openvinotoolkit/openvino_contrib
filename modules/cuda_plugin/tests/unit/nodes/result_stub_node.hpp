// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeinfo>
#include <ngraph/node.hpp>
#include <ngraph/op/result.hpp>

struct ResultStubNode : ngraph::op::Result {
    using ngraph::op::Result::Result;

    inline static constexpr type_info_t type_info{"Result", 0};
    const type_info_t& get_type_info() const override {
        return type_info;
    }

    std::shared_ptr<ngraph::Node>
    clone_with_new_inputs(const ngraph::OutputVector& inputs) const override {
        return std::make_shared<ResultStubNode>();
    }
};
