"""
 Copyright (C) 2018-2023 Intel Corporation

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
from openvino.tools.mo.ops.dft import DFT, IDFT
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.pack import PackOp
from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.ops.squeeze import Squeeze

class RFFT(FrontReplacementOp):
    op = 'RFFT'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        if node.module.inverse:
            axes = Const(graph, {'value': int64_array(range(2, node.module.num_axes - 1))}).create_node()
            dft_node = IDFT(graph, dict(name=node.name, in_ports_count=2)).create_node([node.in_node(0), axes])

            # Slice a real part
            begin_id = Const(graph, {'value': int64_array([0, 0])}).create_node()
            end_id = Const(graph, {'value': int64_array([0, 1])}).create_node()
            real = StridedSlice(graph, dict(name=node.name + '/real',
                                            begin_mask=[0, 0],
                                            end_mask=[0, 1],
                                            shrink_axis_mask=[0, 0],
                                            new_axis_mask=[0],
                                            ellipsis_mask=[1, 0])).create_node([dft_node, begin_id, end_id])

            squeeze_axis = Const(graph, {'value': -1}).create_node()
            res = Squeeze(graph, dict(name=node.name + '/squeeze')).create_node([real, squeeze_axis])

            return [res.id]
        else:
            zero = Const(graph, {'value': 0.0}).create_node()
            imag = Mul(graph, dict(name=node.name + '/imag')).create_node([node.in_node(0), zero])
            cmplx = PackOp(graph, dict(name=node.name + '/complex', axis=-1)).create_node([node.in_node(0), imag])

            axes = Const(graph, {'value': int64_array(range(2, node.module.num_axes))}).create_node()
            dft_node = DFT(graph, dict(name=node.name, in_ports_count=2)).create_node([cmplx, axes])
            return [dft_node.id]
