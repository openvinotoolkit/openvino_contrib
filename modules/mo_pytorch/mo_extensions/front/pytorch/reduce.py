"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.ReduceOps import *
from openvino.tools.mo.ops.const import Const
import numpy as np


class ReduceSumReplacement(FrontReplacementOp):
    op = 'ReduceSum'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axes = np.arange(node.module.num_axes)
        axes = Const(graph, {'value': axes}).create_node()
        red = ReduceSum(graph, dict(name=node.name)).create_node([node.in_node(0), axes])
        return [red.id]


class ReduceMeanReplacement(FrontReplacementOp):
    op = 'ReduceMean'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axis = Const(graph, {'value': node.module.dim}).create_node()
        red = ReduceMean(graph, dict(name=node.name)).create_node([node.in_node(0), axis])
        return [red.id]
