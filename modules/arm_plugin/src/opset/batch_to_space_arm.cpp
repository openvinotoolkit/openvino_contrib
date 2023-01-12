// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_arm.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

ArmPlugin::opset::v1::ArmBatchToSpace::ArmBatchToSpace(const ov::Output<Node> &data,
                                                       const ov::Output<Node> &block_shape,
                                                       const ov::Output<Node> &crops_begin,
                                                       const ov::Output<Node> &crops_end) :
                                                       BatchToSpace({data, block_shape, crops_begin, crops_end}) {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    ov::mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}
