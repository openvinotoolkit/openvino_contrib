"""
 Copyright (C) 2018-2022 Intel Corporation

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
from openvino.tools.mo.ops.pad import Pad
from openvino.tools.mo.ops.const import Const


class Padding(FrontReplacementOp):
    op = 'Padding'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        # Note that PyTorch paddings are reversed
        pads_begin = node.module.pad[::2][::-1]
        pads_end = node.module.pad[1::2][::-1]

        pads_begin = Const(graph, {'name': node.name + '/pads_begin', 'value': pads_begin}).create_node()
        pads_end = Const(graph, {'name': node.name + '/pads_end', 'value': pads_end}).create_node()
        pad_value = Const(graph, {'name': node.name + '/pad_value', 'value': 0.0}).create_node()

        pad = Pad(graph, dict(name=node.name)).create_node([node.in_node(0), pads_begin, pads_end, pad_value])
        return [pad.id]
