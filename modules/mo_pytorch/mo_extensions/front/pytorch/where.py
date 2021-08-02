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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from extensions.ops.select import Select
from mo.ops.const import Const


class Where(FrontReplacementOp):
    op = 'Where'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        cond = Const(graph, {'value': node.module.condition}).create_node()
        else_branch = Const(graph, {'value': node.module.y}).create_node()
        select = Select(graph, dict(name=node.name)).create_node([cond, node.in_node(0), else_branch])
        return [select.id]
