# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from openvino.runtime import Output, op
from openvino.runtime import opset12 as opset


def get_greedy_decoding_ov_subgraph(logits_node: op.Parameter) -> List[Output]:
    argmax = opset.topk(
        data=logits_node,
        k=1,
        axis=-1,
        mode="max",
        sort="none",
        name="ArgMax",
    )
    return opset.squeeze(
        data=argmax.output(1),
        axes=-1,
    ).outputs()
