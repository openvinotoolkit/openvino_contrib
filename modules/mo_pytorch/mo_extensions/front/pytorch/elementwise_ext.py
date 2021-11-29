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

from extensions.ops.elementwise import *
from mo.front.extractor import FrontExtractorOp
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.eltwise_n import EltwiseNAdd, EltwiseNMax
from mo.ops.power import AttributedPower
from extensions.ops.activation_ops import *
from mo.ops.const import Const
from mo.ops.clamp import Clamp
from extensions.ops.Cast import Cast


class AddFrontExtractor(FrontExtractorOp):
    op = 'Add'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Add.update_node_stat(node)
        return cls.enabled


class SubFrontExtractor(FrontExtractorOp):
    op = 'Sub'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Sub.update_node_stat(node)
        return cls.enabled


class MulFrontExtractor(FrontExtractorOp):
    op = 'Mul'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Mul.update_node_stat(node)
        return cls.enabled


class DivFrontExtractor(FrontExtractorOp):
    op = 'Div'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Div.update_node_stat(node)
        return cls.enabled

class AbsFrontExtractor(FrontExtractorOp):
    op = 'Abs'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Abs.update_node_stat(node)
        return cls.enabled

class PowFrontExtractor(FrontExtractorOp):
    op = 'Pow'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = {
            'power': node.module.exponent,
        }
        AttributedPower.update_node_stat(node, attrs)
        return cls.enabled


# log2(x) = ln(x) / ln(2)
class Log2Replacement(FrontReplacementOp):
    op = 'Log2'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        log = Log(graph, dict(name=node.name + '/log')).create_node([node.in_node(0)])
        scale = Const(graph, {'value': np.log(2)}).create_node()
        div = Div(graph, dict(name=node.name + '/scale')).create_node([log, scale])
        return [div.id]


class LessFrontExtractor(FrontExtractorOp):
    op = 'Less'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Less.update_node_stat(node)
        return cls.enabled


class ZerosLike(FrontExtractorOp):
    op = 'ZerosLike'
    enabled = True

    @classmethod
    def extract(cls, node):
        AttributedPower.update_node_stat(node, {'scale': 0})
        return cls.enabled


class SoftPlusOp(FrontExtractorOp):
    op = 'SoftPlus'
    enabled = True

    @classmethod
    def extract(cls, node):
        SoftPlus.update_node_stat(node)
        return cls.enabled


class SqrtFrontExtractor(FrontExtractorOp):
    op = 'Sqrt'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = {
            'power': 0.5,
        }
        AttributedPower.update_node_stat(node, attrs)
        return cls.enabled


class FloorOp(FrontExtractorOp):
    op = 'Floor'
    enabled = True

    @classmethod
    def extract(cls, node):
        Floor.update_node_stat(node)
        return cls.enabled


class EqualOp(FrontExtractorOp):
    op = 'Equal'
    enabled = True

    @classmethod
    def extract(cls, node):
        Equal.update_node_stat(node)
        return cls.enabled


class CastFrontExtractor(FrontExtractorOp):
    op = 'Cast'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = {
            'dst_type': node.module.dst_type,
        }
        Cast.update_node_stat(node, attrs)
        return cls.enabled
