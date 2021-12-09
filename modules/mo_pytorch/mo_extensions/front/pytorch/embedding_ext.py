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

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.const import Const


class EmbeddingExtractor(FrontReplacementOp):
    op = 'Embedding'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        axis = Const(graph, {'value': 0}).create_node()
        inputs = [node.in_node(1),  # weight
                  node.in_node(0),  # input_ids
                  axis]
        gather = Gather(graph, dict(name=node.name)).create_node(inputs)
        return [gather.id]
