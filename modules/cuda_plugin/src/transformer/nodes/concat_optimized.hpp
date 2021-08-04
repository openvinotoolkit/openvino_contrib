// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/concat.hpp>

namespace CUDAPlugin::nodes {

class ConcatOptimized : public ngraph::op::Concat {
   public:
    using ngraph::op::Concat::Concat;

    inline static constexpr type_info_t type_info{"ConcatOptimized", 0};
    const type_info_t& get_type_info() const override { return type_info; }
};

}
