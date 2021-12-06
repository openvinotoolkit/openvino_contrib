from packaging import version
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mo.utils.error import Error

# Callback which is executed after nn.Module forward
def forward_hook(self, inputs, output=None):
    # Get source graph from one of the dynamic inputs
    for inp in inputs:
        graph = inp.graph
        if graph:
            break

    if graph is None:
        raise Error('No graph found')
    layer_type = self.__class__.__name__

    # Create a unique name
    name = graph.unique_id(prefix=layer_type + '_')

    graph.add_node(name, kind='op', op=layer_type, name=name, module=self)

    # Find all inputs
    for idx, inp in enumerate(inputs):
        src_id = inp.node_name
        if src_id is None:
            param_name = name + '/const_' + str(idx)
            inp.node_name = param_name
            graph.add_node(param_name, kind='op', op='Const', value=inp.tensor().numpy())
            edge_attrs = {
                'out': 0,
                'in': idx,
                'name': param_name,
                'fw_tensor_debug_info': [(param_name, param_name)],
                'in_attrs': ['in', 'name'],
                'out_attrs': ['out', 'name'],
                'data_attrs': ['fw_tensor_debug_info']
            }
            graph.add_edge(param_name, name, **edge_attrs)
            continue

        edge_attrs = {
            'out': inp.port_id,
            'in': idx,
            'name': src_id,
            'fw_tensor_debug_info': [(src_id, src_id)],
            'in_attrs': ['in', 'name'],
            'out_attrs': ['out', 'name'],
            'data_attrs': ['fw_tensor_debug_info']
        }
        graph.add_edge(src_id, name, **edge_attrs)

    # state_dict is an OrderedDict that means all the parameterd are
    # ordered by connection
    for idx, (key, value) in enumerate(self.state_dict().items()):
        param_name = name + '/' + key
        graph.add_node(param_name, name=param_name, kind='op', op='Const', value=value.numpy())
        edge_attrs = {
            'out': 0,
            'in': len(inputs) + idx,
            'name': param_name,
            'fw_tensor_debug_info': [(param_name, param_name)],
            'in_attrs': ['in', 'name'],
            'out_attrs': ['out', 'name'],
            'data_attrs': ['fw_tensor_debug_info']
        }
        graph.add_edge(param_name, name, **edge_attrs)

    if isinstance(output, tuple):
        outputs = []
        shapes = self.infer_shapes(inputs)
        if len(shapes) != len(output):
            raise Exception("Incorrect number of output shapes")
        for i, (out, shape) in enumerate(zip(output, shapes)):
            if not isinstance(out, OpenVINOTensor):
                out = OpenVINOTensor(out)

            out.graph = graph
            out.node_name = name
            out.port_id = i
            out.dynamic_shape = shape
            outputs.append(out)
        return tuple(outputs)

    if not isinstance(output, OpenVINOTensor):
        output = OpenVINOTensor(output)
        output.graph = graph
        if hasattr(self, 'infer_shapes'):
            output.dynamic_shape = self.infer_shapes(inputs)
        else:
            output.dynamic_shape = inputs[0].dynamic_shape

    output.node_name = name
    return output

# PyTorch functional ops and Tensor operations are not tracked by forward_hook.
# So we need to introduce own tensor type to track them.
HANDLED_FUNCTIONS = {}
class OpenVINOTensor(object):
    def __init__(self, value=None):
        self._value = value
        self.graph = None
        self.node_name = None
        self.port_id = 0
        self.dynamic_shape = None
        if hasattr(value, 'shape'):
            self.dynamic_shape = list(value.shape) or [1]
        self.requires_grad = False
        self.device = 'cpu'
        self.dtype = torch.float32
        if self.requires_grad:
            raise Error('Model in training mode is used')

    def __repr__(self):
        return self.node_name

    def tensor(self):
        return self._value

    def numel(self):
        return self

    def dim(self):
        return len(self.dynamic_shape)

    @property
    def shape(self):
        return tuple(self.dynamic_shape)

    def size(self, dim=None):
        return self.shape if dim is None else self.dynamic_shape[dim]

    def clone(self):
        class Identity(nn.Module):
            def __init__(self):
                super().__init__()

        return forward_hook(Identity(), (self,))

    def to(self, dtype):
        mapping = {
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.float32: np.float32,
        }
        if not dtype in mapping:
            raise Exception("Unexpected dst type:", dtype)

        class Cast(nn.Module):
            def __init__(self):
                super().__init__()
                self.dst_type = mapping[dtype]

        return forward_hook(Cast(), (self,))

    def split(self, split_size, dim):
        num_splits = self.dynamic_shape[dim] // split_size

        class Split(nn.Module):
            def __init__(self, num_splits, dim):
                super().__init__()
                self.num_splits = num_splits
                self.dim = dim

            def infer_shapes(self, inputs):
                shape = list(inputs[0].dynamic_shape)
                shape[dim] = shape[dim] // num_splits
                return [shape] * num_splits

        outputs = tuple([OpenVINOTensor() for _ in range(num_splits)])
        return forward_hook(Split(num_splits, dim), (self,), outputs)

    # Overrides += over tensors
    def __iadd__(self, a):
        return self + a

    def __add__(self, a):
        if isinstance(a, OpenVINOTensor):
            class Add(nn.Module):
                pass
            return forward_hook(Add(), (self, a))
        else:
            class Add(nn.Module):
                def __init__(self, value):
                    super().__init__()
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    self.register_buffer('add', value)

            return forward_hook(Add(a), (self,))

    def __sub__(self, a):
        if isinstance(a, OpenVINOTensor):
            class Sub(nn.Module):
                pass
            return forward_hook(Sub(), (self, a))
        else:
            class Sub(nn.Module):
                def __init__(self, value):
                    super().__init__()
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    self.register_buffer('sub', value)

            return forward_hook(Sub(a), (self,))

    def __rsub__(self, a):
        a = a if isinstance(a, OpenVINOTensor) else OpenVINOTensor(torch.tensor(a))
        class Sub(nn.Module):
            pass
        return forward_hook(Sub(), (a, self))

    def __radd__(self, a):
        if isinstance(a, OpenVINOTensor):
            class Add(nn.Module):
                pass
            return forward_hook(Add(), (self, a))
        else:
            class Add(nn.Module):
                def __init__(self, value):
                    super().__init__()
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    self.register_buffer('add', value)

            return forward_hook(Add(a), (self,))

    def __eq__(self, a):
        class Equal(nn.Module):
            def __init__(self, value):
                super().__init__()

        return forward_hook(Equal(a), (self, OpenVINOTensor(torch.tensor(a))))

    def __getitem__(self, key):
        begin_id = []
        end_id = []
        begin_mask = []
        end_mask = []
        shrink_axis_mask = []
        ellipsis_mask = []

        if not isinstance(key, tuple):
            key = (key,)

        if not isinstance(key, tuple):
            key = (key,)

        for item in key:
            if isinstance(item, int):
                begin_id.append(item)
                end_id.append(item + 1)

                shrink_axis_mask.append(1)
                begin_mask.append(1)
                end_mask.append(1)
                ellipsis_mask.append(0)

            elif isinstance(item, slice):
                begin_id.append(item.start if item.start else 0)
                begin_mask.append(1 if item.start else 0)

                end_id.append(item.stop if item.stop else 0)
                end_mask.append(1 if item.stop else 0)

                shrink_axis_mask.append(0)
                ellipsis_mask.append(0)

            elif str(type(item)) == "<class 'ellipsis'>":
                begin_id.append(0)
                end_id.append(0)
                shrink_axis_mask.append(0)
                begin_mask.append(0)
                end_mask.append(0)
                ellipsis_mask.append(1)

        class StridedSlice(nn.Module):
            def __init__(self, begin, end, begin_mask, end_mask, shrink_mask, ellipsis_mask):
                super().__init__()
                self.begin_mask = begin_mask
                self.end_mask = end_mask
                self.shrink_axis_mask = shrink_mask
                self.ellipsis_mask = ellipsis_mask
                self.register_buffer('begin_id', torch.tensor(begin))
                self.register_buffer('end_id', torch.tensor(end))

            def infer_shapes(self, inputs):
                def canonical(dim, size):
                    return dim if dim >= 0 else size + dim

                axis = 0
                shape = list(inputs[0].dynamic_shape)
                for i in range(len(shrink_axis_mask)):
                    if shrink_axis_mask[i]:
                        del shape[axis]
                        continue

                    if ellipsis_mask[i]:
                        axis = i - len(ellipsis_mask) + 1
                        continue

                    dim = shape[axis]
                    begin = canonical(begin_id[i], dim) if begin_mask[i] else 0
                    end = canonical(end_id[i], dim) if end_mask[i] else dim
                    shape[axis] = end - begin
                    axis += 1

                return shape if shape else [1]

        sslice = StridedSlice(begin_id, end_id, begin_mask, end_mask, shrink_axis_mask, ellipsis_mask)
        return forward_hook(sslice, (self,))

    # a * value
    def __mul__(self, a):
        class Mul(nn.Module):
            def __init__(self, value=None):
                super().__init__()
                if value is not None:
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    self.register_buffer('mul', value)

            def infer_shapes(self, inputs):
                if len(inputs) == 1 or np.prod(inputs[0].dynamic_shape) > np.prod(inputs[1].dynamic_shape):
                    return inputs[0].dynamic_shape
                else:
                    return inputs[1].dynamic_shape

        if isinstance(a, OpenVINOTensor):
            return forward_hook(Mul(), (self, a))
        else:
            return forward_hook(Mul(a), (self,))

    def __rmul__(self, a):
        return self * a

    def __pow__(self, exponent):
        class Pow(nn.Module):
            def __init__(self):
                super().__init__()
                self.exponent = exponent

        return forward_hook(Pow(), (self,))

    def __truediv__(self, a):
        if isinstance(a, OpenVINOTensor):
            class Div(nn.Module):
                pass
            return forward_hook(Div(), (self, a))
        else:
            class Div(nn.Module):
                def __init__(self, value):
                    super().__init__()
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    self.register_buffer('div', value)

            return forward_hook(Div(a), (self,))

    def view(self, *shape):
        class Reshape(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = list(np.array(shape).flat)

            def infer_shapes(self, inputs):
                target_shape = list(self.shape)  # Create a copy of shape and replace -1
                for i, val in enumerate(self.shape):
                    if val == -1:
                        target_shape[i] = np.prod(inputs[0].dynamic_shape) // abs(np.prod(target_shape))
                        break
                return target_shape

        return forward_hook(Reshape(shape), (self,))


    def flatten(self, *args, **kwargs):
        return torch.flatten(self, *args, **kwargs)


    def softmax(self, dim):
        class Softmax(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

        return forward_hook(Softmax(dim), (self,))


    def reshape(self, *shape):
        return self.view(*shape)

    def expand_as(self, other):
        return self.view(*other.shape)

    def permute(self, *order):
        class Transpose(nn.Module):
            def __init__(self, order):
                super().__init__()
                self.order = order

            def infer_shapes(self, inputs):
                return [inputs[0].dynamic_shape[i] for i in self.order]

        return forward_hook(Transpose(order), (self,))

    def repeat(self, *repeats):
        repeats = OpenVINOTensor(torch.tensor(repeats))

        class Tile(nn.Module):
            def __init__(self):
                super().__init__()

        return forward_hook(Tile(), (self, repeats))

    def unsqueeze(self, dim):
        class Unsqueeze(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('unsqueeze_dims', torch.tensor(dim))

            def infer_shapes(self, inputs):
                shape = list(inputs[0].dynamic_shape)
                shape.insert(1, dim if dim >= 0 else (len(shape) + dim + 1))
                return shape

        return forward_hook(Unsqueeze(), (self,))

    def transpose(self, dim0, dim1):
        order = [i for i in range(self.dim())]
        order[dim0], order[dim1] = order[dim1], order[dim0]
        return self.permute(*order)

    def __lt__(self, other):
        class Less(nn.Module):
            def __init__(self):
                super().__init__()

        return forward_hook(Less(), (self, other))

    def sigmoid(self):
        class Sigmoid(nn.Module):
            def __init__(self):
                super().__init__()

        return forward_hook(Sigmoid(), (self,))

    def contiguous(self):
        return self

    def is_floating_point(self):
        return True

    def type(self, dtype):
        return self

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor, OpenVINOTensor))
            for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


import functools
def implements(torch_function):
    """Register a torch function override for OpenVINOTensor"""
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func
    return decorator


@implements(F.dropout)
@implements(F.dropout3d)
def function_hook(input, *args, **kwargs):
    return input


@implements(F.max_pool2d)
def function_hook(input, *args, **kwargs):

    class MaxPool2d(nn.Module):
        def __init__(self, kernel_size, stride, padding, dilation, return_indices, ceil_mode):
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.return_indices = return_indices
            self.ceil_mode = ceil_mode

        def infer_shapes(self, inputs):
            shape = list(inputs[0].dynamic_shape)
            for i in range(2):
                p = self.padding if isinstance(self.padding, int) else self.padding[i]
                k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[i]
                d = self.dilation if isinstance(self.dilation, int) else self.dilation[i]
                s = self.stride if isinstance(self.stride, int) else self.stride[i]
                shape[2 + i] = 1 + (shape[2 + i] + 2 * p - d * (k - 1) - 1) // s
            return shape

    return forward_hook(MaxPool2d(*args, **kwargs), (input,))


@implements(F.avg_pool2d)
def function_hook(input, *args, **kwargs):
    class AvgPool2d(nn.Module):
        def __init__(self, kernel_size, stride, padding):
            super().__init__()
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding

    return forward_hook(AvgPool2d(*args, **kwargs), (input,))


@implements(F.adaptive_avg_pool2d)
def function_hook(input, output_size):
    class AdaptiveAvgPool2d(nn.Module):
        def __init__(self):
            super().__init__()
            self.output_size = output_size

    return forward_hook(AdaptiveAvgPool2d(), (input,))


@implements(torch.relu_)
def function_hook(input, *args, **kwargs):

    class ReLU(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(ReLU(*args, **kwargs), (input,))


@implements(torch.tanh)
def function_hook(input, *args, **kwargs):

    class Tanh(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Tanh(*args, **kwargs), (input,))


@implements(torch.pow)
def function_hook(input, *args, **kwargs):

    class Pow(nn.Module):
        def __init__(self, exponent):
            super().__init__()
            self.exponent = exponent

    return forward_hook(Pow(*args, **kwargs), (input,))

@implements(torch.sqrt)
def function_hook(input, *args, **kwargs):

    class Sqrt(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Sqrt(*args, **kwargs), (input,))

@implements(torch.floor)
def function_hook(input, *args, **kwargs):

    class Floor(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Floor(*args, **kwargs), (input,))

@implements(torch.clamp)
def function_hook(input, min, max):

    class Clamp(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Clamp(), (input, OpenVINOTensor(torch.tensor(min)), OpenVINOTensor(torch.tensor(max))))

@implements(torch.unsqueeze)
def unsqueeze(input, dim):
    return input.unsqueeze(dim)


@implements(F.relu)
def function_hook(input, *args, **kwargs):

    class ReLU(nn.Module):
        def __init__(self, inplace):
            super().__init__()

    return forward_hook(ReLU(*args, **kwargs), (input,))


@implements(torch.sigmoid)
def function_hook(input, *args, **kwargs):

    class Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Sigmoid(*args, **kwargs), (input,))


@implements(F.softmax)
def function_hook(input, dim, _stacklevel, dtype):

    class Softmax(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

    return forward_hook(Softmax(dim), (input,))


@implements(F.leaky_relu)
def function_hook(input, *args, **kwargs):

    class LeakyReLU(nn.Module):
        def __init__(self, negative_slope, inplace):
            super().__init__()
            self.negative_slope = negative_slope

    return forward_hook(LeakyReLU(*args, **kwargs), (input,))


@implements(F.batch_norm)
def function_hook(input, *args, **kwargs):

    class BatchNorm2d(nn.BatchNorm2d):
        def __init__(self, running_mean, running_var, weight, bias, training, momentum, eps):
            if training:
                raise Error('BatchNorm2d in training mode is not implemented')
            super().__init__(num_features=weight.shape[0],
                             momentum=momentum,
                             eps=eps)
            self.load_state_dict({
                'running_mean': running_mean,
                'running_var': running_var,
                'weight': weight,
                'bias': bias,
            })

    return forward_hook(BatchNorm2d(*args, **kwargs), (input,))


@implements(torch.conv2d)
@implements(torch.conv3d)
def function_hook(input, weight, bias, *args, **kwargs):

    base = nn.Conv2d if input.dim() == 4 else nn.Conv3d

    class Convolution(base):
        def __init__(self, weight, bias, stride, padding, dilation, groups):
            super().__init__(in_channels=weight.shape[1] * groups,
                             out_channels=weight.shape[0],
                             kernel_size=weight.shape[2:],
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=not bias is None)
            params = {'weight': weight}
            if not bias is None:
                params['bias'] = bias
            self.load_state_dict(params)

        def infer_shapes(self, inputs):
            shape = list(inputs[0].dynamic_shape)
            shape[1] = self.out_channels
            for i in range(input.dim() - 2):
                p = self.padding if isinstance(self.padding, int) else self.padding[i]
                k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[i]
                d = self.dilation if isinstance(self.dilation, int) else self.dilation[i]
                s = self.stride if isinstance(self.stride, int) else self.stride[i]
                shape[2 + i] = 1 + (shape[2 + i] + 2 * p - d * (k - 1) - 1) // s
            return shape

    return forward_hook(Convolution(weight, bias, *args, **kwargs), (input,))


@implements(torch.conv_transpose2d)
def function_hook(input, weight, bias, *args, **kwargs):
    class Deconvolution(nn.ConvTranspose2d):
        def __init__(self, weight, bias, stride, padding, output_padding, groups, dilation):
            super().__init__(in_channels=input.shape[1],
                             out_channels=weight.shape[1] * groups,
                             kernel_size=weight.shape[2:],
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=not bias is None)
            params = {'weight': weight}
            if not bias is None:
                params['bias'] = bias
            self.load_state_dict(params)


        def infer_shapes(self, inputs):
            shape = list(inputs[0].dynamic_shape)
            shape[1] = self.out_channels
            for i in range(input.dim() - 2):
                p = self.padding if isinstance(self.padding, int) else self.padding[i]
                k = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[i]
                d = self.dilation if isinstance(self.dilation, int) else self.dilation[i]
                s = self.stride if isinstance(self.stride, int) else self.stride[i]
                shape[2 + i] = s * (shape[2 + i] - 1) + k - 2 * p
            return shape

    return forward_hook(Deconvolution(weight, bias, *args, **kwargs), (input,))


@implements(torch.flatten)
def function_hook(input, start_dim, end_dim=-1):
    class Flatten(nn.Module):
        def __init__(self):
            super().__init__()
            self.axis = start_dim
            self.end_axis = end_dim

        def infer_shapes(self, inputs):
            shape = inputs[0].dynamic_shape
            end = self.end_axis if self.end_axis >= 0 else (len(shape) + self.end_axis + 1)

            out = inputs[0].dynamic_shape[:self.axis]
            out += [np.prod(inputs[0].dynamic_shape[self.axis : end])]
            out += inputs[0].dynamic_shape[end:]
            return out

    return forward_hook(Flatten(), (input,))


@implements(F.instance_norm)
def function_hook(input, *args, **kwargs):
    class InstanceNorm(nn.Module):
        def __init__(self, running_mean, running_var, weight, bias, use_input_stats, momentum, eps):
            super().__init__()
            self.running_mean = running_mean
            self.running_var = running_var
            self.weight = weight
            self.bias = bias
            self.use_input_stats = use_input_stats
            self.momentum = momentum
            self.eps = eps
            self.dims = input.dim()

    return forward_hook(InstanceNorm(*args, **kwargs), (input,))


@implements(F.interpolate)
def function_hook(input, *args, **kwargs):

    class Upsample(nn.Module):
        def __init__(self, size, scale_factor, mode, align_corners, recompute_scale_factor):
            super().__init__()
            self.size = size
            self.scale_factor = scale_factor
            self.mode = mode
            self.align_corners = align_corners
            self.recompute_scale_factor = recompute_scale_factor
            self.dims = input.dim()

        def infer_shapes(self, inputs):
            if self.size is not None:
                return inputs[0].dynamic_shape[:2] + list(self.size)
            if self.scale_factor:
                shape = list(inputs[0].dynamic_shape)
                for i in range(2, len(shape)):
                    shape[i] = int(shape[i] * self.scale_factor)
                return shape

    return forward_hook(Upsample(*args, **kwargs), (input,))


# x - value
@implements(torch.rsub)
def function_hook(value, x):
    class Sub(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer('sub', value)

    return forward_hook(Sub(value), (x,))


@implements(torch.cat)
def function_hook(inputs, dim=0):
    class Concat(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def infer_shapes(self, inputs):
            shape = list(inputs[0].dynamic_shape)
            if shape[self.dim] == -1:
                return shape

            for inp in inputs[1:]:
                if inp.dynamic_shape[self.dim] == -1:
                    shape[self.dim] = -1
                    return shape

                shape[self.dim] += inp.dynamic_shape[self.dim]

            return shape

    inputs = [inp if isinstance(inp, OpenVINOTensor) else OpenVINOTensor(inp) for inp in inputs]
    return forward_hook(Concat(dim), inputs)


@implements(torch.embedding)
def function_hook(weight, input, *args, **kwargs):

    class Embedding(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.register_buffer('weight', weight)

        def infer_shapes(self, inputs):
            return inputs[0].dynamic_shape + [weight.shape[1]]

    return forward_hook(Embedding(weight), (input,))


@implements(F.embedding)
def function_hook(input, weight, *args, **kwargs):
    return torch.embedding(weight, input)


@implements(F.layer_norm)
def function_hook(input, normalized_shape, weight, bias, eps):

    class LayerNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('weight', weight)
            self.register_buffer('bias', bias)
            self.eps = eps

    return forward_hook(LayerNorm(), (input,))


@implements(torch.addmm)
def function_hook(bias, mat1, mat2):

    class ADDMM(nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.register_buffer('weight', weight)
            self.register_buffer('bias', bias)

        def infer_shapes(self, inputs):
            return inputs[0].dynamic_shape[:-1] + [self.state_dict()['weight'].shape[1]]

    return forward_hook(ADDMM(mat2, bias), (mat1,))


@implements(F.linear)
def function_hook(input, weight, bias=None):
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()

        def infer_shapes(self, inputs):
            return inputs[0].dynamic_shape[:-1] + [inputs[1].dynamic_shape[0]]

    weight = OpenVINOTensor(weight)
    if bias is not None:
        bias = OpenVINOTensor(bias)
        return forward_hook(Linear(), (input, weight, bias))
    else:
        return forward_hook(Linear(), (input, weight))


@implements(torch.stack)
def function_hook(inputs, dim=0):
    class Stack(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def infer_shapes(self, inputs):
            dim = self.dim if self.dim >= 0 else (inputs[0].dim() + self.dim + 1)
            shape = inputs[0].dynamic_shape
            shape.insert(dim, len(inputs))
            return shape

    inputs = [inp if isinstance(inp, OpenVINOTensor) else OpenVINOTensor(inp) for inp in inputs]
    return forward_hook(Stack(dim), inputs)


@implements(torch.matmul)
def function_hook(mat0, mat1):

    class Matmul(nn.Module):
        def __init__(self):
            super().__init__()

        def infer_shapes(self, inputs):
            return inputs[0].dynamic_shape[:-1] + [inputs[1].dynamic_shape[-1]]

    return forward_hook(Matmul(), (mat0, mat1))


@implements(torch.where)
def function_hook(condition, x, y):

    class Select(nn.Module):
        def __init__(self):
            super().__init__()

        def infer_shapes(self, inputs):
            return inputs[1].dynamic_shape or inputs[2].dynamic_shape

    inputs = [condition, x, y]
    inputs = [inp if isinstance(inp, OpenVINOTensor) else OpenVINOTensor(inp) for inp in inputs]
    return forward_hook(Select(), inputs)


@implements(torch.rfft if version.parse(torch.__version__) < version.parse('1.8.0') else None)
def function_hook(x, *args, **kwargs):

    class RFFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.inverse = False
            self.num_axes = x.dim()

        def infer_shapes(self, inputs):
            return inputs[0].dynamic_shape + [2]

    return forward_hook(RFFT(), (x,))


@implements(torch.irfft if version.parse(torch.__version__) < version.parse('1.8.0') else None)
def function_hook(x, *args, **kwargs):

    class RFFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.inverse = True
            self.num_axes = x.dim()

        def infer_shapes(self, inputs):
            return inputs[0].dynamic_shape[:-1]

    return forward_hook(RFFT(), (x,))

@implements(torch.Tensor.type_as)
def function_hook(src, dst):
    class ConvertLike(nn.Module):
        def __init__(self):
            super().__init__()

    inp = OpenVINOTensor(src)
    inp.graph = dst.graph
    inp.dynamic_shape = list(src.shape)

    return forward_hook(ConvertLike(), (inp, dst))


@implements(torch.Tensor.copy_)
def function_hook(dst, src):
    return dst


@implements(F._pad)
def function_hook(input, pad, mode, value):

    class Padding(nn.Module):
        def __init__(self, pad):
            super().__init__()
            self.pad = pad

        def infer_shapes(self, inputs):
            shape = list(inputs[0].dynamic_shape)
            # Note that paddings are reversed
            for i in range(len(shape)):
                shape[-1 - i] += self.pad[2 * i] + self.pad[2 * i + 1]
            return shape

    if any([value != 0 for value in pad]):
        return forward_hook(Padding(pad), (input,))
    else:
        return input

@implements(torch.roll)
def function_hook(input, shifts, dims):
    class Roll(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('shifts', torch.tensor(shifts))
            self.register_buffer('dims', torch.tensor(dims))

    return forward_hook(Roll(), (input,))

@implements(torch.log2)
def function_hook(input, *args, **kwargs):

    class Log2(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Log2(), (input,))

@implements(torch.sum)
def function_hook(input, *args, **kwargs):

    class ReduceSum(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_axes = input.dim()

        def infer_shapes(self, inputs):
            return [1]

    return forward_hook(ReduceSum(), (input,))

@implements(torch.mean)
def function_hook(input, *args, **kwargs):

    class ReduceMean(nn.Module):
        def __init__(self, dim, keepdim):
            super().__init__()
            self.dim = dim
            self.keepdim = keepdim

        def infer_shapes(self, inputs):
            shape = list(inputs[0].dynamic_shape)
            if self.keepdim:
                shape[self.dim] = 1
            else:
                del shape[self.dim]
            return shape

    return forward_hook(ReduceMean(*args, **kwargs), (input,))


@implements(torch.abs)
def function_hook(input, *args, **kwargs):

    class Abs(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(Abs(*args, **kwargs), (input,))


@implements(torch.zeros_like)
def function_hook(input):

    class ZerosLike(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(ZerosLike(), (input,))


@implements(F.softplus)
def function_hook(input, *args, **kwargs):

    class SoftPlus(nn.Module):
        def __init__(self):
            super().__init__()

    return forward_hook(SoftPlus(), (input,))

@implements(torch.chunk)
def function_hook(input, num_chunks, dim):
    split_size = input.dynamic_shape[dim] // num_chunks
    return input.split(split_size, dim)


@implements(torch.topk)
def function_hook(input, k, dim=0):

    class TopK(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = dim

        def infer_shapes(self, inputs):
            shape = inputs[0].dynamic_shape
            out = shape[:dim] + [k] + shape[dim + 1:]
            return out, out

    outputs = (OpenVINOTensor(), OpenVINOTensor())
    return forward_hook(TopK(), (input, OpenVINOTensor(torch.tensor(k))), outputs)


@implements(torch.gather)
def function_hook(input, dim, index):

    class Gather(nn.Module):
        def __init__(self):
            super().__init__()

        def infer_shapes(self, inputs):
            out = inputs[0].dynamic_shape[:dim]
            out += inputs[1].dynamic_shape
            out += inputs[0].dynamic_shape[dim + 1:]
            return out

    axes = OpenVINOTensor(torch.tensor(dim))
    return forward_hook(Gather(), (input, index, axes))
