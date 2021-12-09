"""
 Copyright (C) 2018-2020 Intel Corporation

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
import numpy as np

from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.MatMul import MatMul
from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.ops.const import Const


class Linear(FrontReplacementOp):
    op = 'Linear'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        matmul = MatMul(graph, dict(name=node.name, transpose_b=True)).create_node([node.in_node(0), node.in_node(1)])

        # Bias
        if len(node.in_nodes()) > 2:
            matmul = Add(graph, dict(name=node.name + '/bias')).create_node([matmul, node.in_node(2)])

        return [matmul.id]


class ADDMM(FrontReplacementOp):
    op = 'ADDMM'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        weight = node.module.weight.detach().numpy()
        bias = node.module.bias.detach().numpy()

        weight = Const(graph, {'value': weight}).create_node()
        bias = Const(graph, {'value': bias}).create_node()
        matmul = MatMul(graph, dict(name=node.name)).create_node([node.in_node(0), weight])
        matmul = Add(graph, dict(name=node.name + '/bias')).create_node([matmul, bias])
        return [matmul.id]
