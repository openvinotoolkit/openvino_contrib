#!/usr/bin/env python3
import argparse
import glob
import json

from json import JSONEncoder
from bs4 import BeautifulSoup as bs


def get_arguments():
    parser = argparse.ArgumentParser(description='Process OpenVINO *.xml model')
    parser.add_argument('--models',
                        required=True,
                        type=str,
                        nargs='*',
                        help='Path to model')
    parser.add_argument('--ops',
                        metavar='OPS',
                        type=str,
                        default='all',
                        nargs='*',
                        help='Which operations to search')
    return parser.parse_args()


class Operation:
    def __init__(self, name):
        self.name = name
        self.shapes = set()


class OperationPort:
    def __init__(self, shape, attrs):
        self.shape = shape
        self.attrs = attrs
        del self.attrs['id']

    def __eq__(self, other):
        return self.shape.__eq__(other.shape)

    def __hash__(self):
        return hash(self.shape)


class OperationShape:
    def __init__(self, version, attrs, inputs, outputs):
        self.version = version
        self.attrs = attrs
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    def __eq__(self, other):
        return self.version.__eq__(other.version) and \
               self.attrs.__eq__(other.attrs) and \
               self.inputs.__eq__(other.inputs) and \
               self.outputs.__eq__(other.outputs)

    def __hash__(self):
        return hash(self.version) ^ hash(json.dumps(self.attrs)) ^ hash(self.inputs) ^ hash(self.outputs)


def parse_input_shapes(layer):
    input_shapes = []
    start_input = False
    for l in layer:
        if l.name == 'input':
            start_input = True
        if start_input and l.name == 'port':
            dims = l.find_all("dim")
            input_shapes.append(OperationPort(tuple(int(dim.text) for dim in dims), l.attrs))
        if l.name == 'output':
            start_input = False
    return input_shapes


def parse_output_shapes(layer):
    output_shapes = []
    output_tag = layer.find("output")
    if output_tag:
        outputs = output_tag.find_all("port")
        for out in outputs:
            dims = out.find_all("dim")
            output_shapes.append(OperationPort(tuple(int(dim.text) for dim in dims), out.attrs))
    return output_shapes


class OperationEncoder(JSONEncoder):
    @staticmethod
    def to_cpp_init_list(shape):
        return f"{shape}".replace("(", "{").replace(")", "}")

    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, OperationShape):
            return {
                'version': obj.version,
                'attrs': obj.attrs,
                'inputs': obj.inputs,
                'outputs': obj.outputs,
            }
        elif isinstance(obj, OperationPort):
            shape = OperationEncoder.to_cpp_init_list(obj.shape)
            if obj.attrs:
                return {
                    'shape': f"{shape}",
                    'attrs': obj.attrs,
                }
            else:
                return f"{shape}"


if __name__ == '__main__':
    args = get_arguments()
    operations = {}
    for model in args.models:
        for filename in glob.glob(model):
            with open(filename, "r") as file:
                content = file.readlines()
                content = "".join(content)
                bs_content = bs(content, "lxml")
                layers = bs_content.find_all("layer")
                layers = sorted(layers, key=lambda l: l.attrs['type'])
                for layer in layers:
                    layer_type = layer.attrs['type']
                    layer_version = layer.attrs['version']
                    if layer_type not in operations:
                        operations[layer_type] = Operation(layer_type)
                    data_tag = layer.find("data")
                    attrs = data_tag.attrs if data_tag else {}
                    input_shapes = parse_input_shapes(layer)
                    output_shapes = parse_output_shapes(layer)
                    operations[layer_type].shapes.add(OperationShape(layer_version, attrs, input_shapes, output_shapes))

    if args.ops == 'all' or args.ops == ['all']:
        for op_name, op in operations.items():
            print(f"{json.dumps(op.__dict__, indent=4, cls=OperationEncoder)}\n")
    else:
        for op in args.ops:
            print(f"{json.dumps(operations[op].__dict__, indent=4, cls=OperationEncoder)}\n")
