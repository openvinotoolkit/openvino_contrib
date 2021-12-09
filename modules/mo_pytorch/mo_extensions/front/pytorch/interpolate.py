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
from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.ops.upsample import UpsampleOp
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.utils.error import Error

class InterpolateReplacer(FrontReplacementOp):
    op = 'Upsample'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        mode = node.module.mode
        if mode.endswith('linear'):  # like bilinear or trilinear
            mode = 'linear'
        align_corners = node.module.align_corners

        if mode == 'linear':
            height = node.module.size[0] if node.module.size is not None else -1
            width = node.module.size[1] if node.module.size is not None else -1
            dims = node.module.dims
            axes = np.arange(2, dims)
            pads = np.zeros(dims, dtype=np.int32)
            scales = np.repeat(node.module.scale_factor, dims - 2).astype(np.float32)
            attrs = {
                'name': node.name,
                'version': 'opset4',
                'height': height,
                'width': width,
                'mode': mode,
                'axes': axes,
                'pads_begin': pads,
                'pads_end': pads,
                'coordinate_transformation_mode': 'align_corners' if align_corners else 'half_pixel',
                'shape_calculation_mode': 'sizes' if node.module.size is not None else 'scales',
            }

            sizes = Const(graph, {'value': np.array([height, width])}).create_node()
            axes = Const(graph, {'value': axes}).create_node()
            scales = Const(graph, {'value': scales}).create_node()
            interp = Interpolate(graph, attrs).create_node([node.in_node(0), sizes, scales, axes])
        else:
            if node.module.size:
                attrs = {
                    'name': node.name,
                    'version': 'opset1',
                    'height': node.module.size[0],
                    'width': node.module.size[1],
                    'mode': mode,
                    'axes': [2, 3],
                    'align_corners': node.module.align_corners,
                }
                interp = Interpolate(graph, attrs).create_node([node.in_node(0)])
            else:
                if not node.module.scale_factor:
                    raise Error('No scale_factor found')
                attrs = {
                    'name': node.name,
                    'height_scale': np.float(node.module.scale_factor),
                    'width_scale': np.float(node.module.scale_factor),
                    'mode': mode,
                    'align_corners': node.module.align_corners,
                }
                interp = UpsampleOp(graph, attrs).create_node([node.in_node(0)])

        return [interp.id]
