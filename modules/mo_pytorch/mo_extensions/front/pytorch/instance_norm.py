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
from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.ops.mvn import MVN
from openvino.tools.mo.ops.const import Const
import numpy as np
from openvino.tools.mo.front.common.partial_infer.utils import int64_array


class InstanceNorm(FrontReplacementOp):
    op = 'InstanceNorm'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        mean = node.module.running_mean.detach().numpy()
        var = node.module.running_var.detach().numpy()
        weight = node.module.weight.detach().numpy()
        bias = node.module.bias.detach().numpy()

        w = weight / np.sqrt(var + node.module.eps)
        b = bias - w * mean

        shape = np.ones(node.module.dims, dtype=np.int32)
        shape[1] = -1  # channels

        w = Const(graph, {'value': w.reshape(shape)}).create_node()
        b = Const(graph, {'value': b.reshape(shape)}).create_node()
        mul = Mul(graph, dict(name=node.name + '/mul')).create_node([node.in_node(0), w])
        add = Add(graph, dict(name=node.name + '/add')).create_node([mul, b])
        return [add.id]


class LayerNormReplace(FrontReplacementOp):
    op = 'LayerNorm'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axis = Const(graph, {'value': int64_array([-1])}).create_node()
        mvn = MVN(graph, dict(name=node.name + '/mvn',
                              eps=node.module.eps,
                              normalize_variance=True,
                              eps_mode='inside_sqrt')).create_node([node.in_node(0), axis])

        weight = node.module.weight.detach().numpy()
        bias = node.module.bias.detach().numpy()

        w = Const(graph, {'value': weight}).create_node()
        b = Const(graph, {'value': bias}).create_node()
        mul = Mul(graph, dict(name=node.name + '/mul')).create_node([mvn, w])
        add = Add(graph, dict(name=node.name + '/add')).create_node([mul, b])
        return [add.id]
