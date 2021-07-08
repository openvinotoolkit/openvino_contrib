from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.op import Op
from extensions.ops.mvn import MVN
from extensions.ops.elementwise import Mul, Add
from mo.ops.const import Const
import numpy as np
from .batchnorm import BatchNorm

class InstanceNorm3d(FrontReplacementOp):
    op = 'InstanceNorm'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        print('~~~~~~~~~~~~~~ 12312312')
        print(node.module.eps)
        # attrs = {
        #     'name': node.name + '/mvn',
        #     'version': 'opset6',
        #     'eps': node.module.eps,
        #     'eps_mode': 'inside_sqrt',
        #     'normalize_variance': True,
        # }
        # axes = Const(graph, {'value': np.array([2, 3, 4])}).create_node()
        # mvn = MVN(graph, attrs).create_node([node.in_node(0), axes])

        # weight = node.module.weight.detach().numpy()
        # weight = weight.reshape(1, -1, 1, 1, 1)
        # weight = Const(graph, {'value': weight}).create_node()
        # mul = Mul(graph, dict(name=node.name + '/mul')).create_node([mvn, weight])

        # bias = node.module.bias.detach().numpy()
        # bias = bias.reshape(1, -1, 1, 1, 1)
        # bias = Const(graph, {'value': bias}).create_node()
        # add = Add(graph, dict(name=node.name + '/add')).create_node([mul, bias])

        # num_features = node.module.weight.shape[0]
        # mean = node.module.running_mean.detach().numpy()
        # std = np.sqrt(node.module.running_var.detach().numpy())
        # weight = node.module.weight.detach().numpy()
        # bias = node.module.bias.detach().numpy()
        # print(std)

        # inputs = [node.in_node(0)]
        # inputs.append(Const(graph, {'value': mean}).create_node())
        # inputs.append(Const(graph, {'value': std}).create_node())
        # inputs.append(Const(graph, {'value': weight}).create_node())
        # inputs.append(Const(graph, {'value': bias}).create_node())
        # bn = BatchNorm(graph, dict(name=node.name, eps=node.module.eps)).create_node(inputs)


        # weight = node.module.weight.detach().numpy()
        # weight = weight.reshape(1, -1, 1, 1, 1)
        # weight = Const(graph, {'value': weight}).create_node()
        # mul = Mul(graph, dict(name=node.name + '/mul')).create_node([mvn, weight])

        # bias = node.module.bias.detach().numpy()
        # bias = bias.reshape(1, -1, 1, 1, 1)
        # bias = Const(graph, {'value': bias}).create_node()
        # add = Add(graph, dict(name=node.name + '/add')).create_node([mul, bias])

        mean = node.module.running_mean.detach().numpy()
        var = node.module.running_var.detach().numpy()
        weight = node.module.weight.detach().numpy()
        bias = node.module.bias.detach().numpy()

        w = weight / np.sqrt(var + node.module.eps)
        b = bias - w * mean

        w = Const(graph, {'value': w.reshape(1, -1, 1, 1, 1)}).create_node()
        b = Const(graph, {'value': b.reshape(1, -1, 1, 1, 1)}).create_node()
        mul = Mul(graph, dict(name=node.name + '/mul')).create_node([node.in_node(0), w])
        add = Add(graph, dict(name=node.name + '/add')).create_node([mul, b])

        return [add.id]

        # inputs = [node.in_node(i) for i in range(5)]
        # bn = BatchNorm(graph, dict(name=node.name, eps=node.module.eps)).create_node(inputs)
        # return [bn.id]
