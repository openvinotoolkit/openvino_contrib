#!/usr/bin/env python3
import argparse

from bs4 import BeautifulSoup as bs


def get_arguments():
    parser = argparse.ArgumentParser(description='Process OpenVINO *.xml model')
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        help='Path to model')
    parser.add_argument('--ops',
                        metavar='OPS',
                        type=str,
                        default='all',
                        nargs='*',
                        help='Which operations to search')
    return parser.parse_args()


class Operation:
    def __init__(self, name, version):
        self.name = name
        self.version = version
        self.shapes = set()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        msg = f"Operation: {self.name}, version {self.version} {{\n"
        msg += f"\tshapes: {self.shapes}\n}}"
        return msg.replace(')}', ')\n\t}')


class OperationShape:
    def __init__(self, attrs, inputs, outputs):
        self.attrs = attrs
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

    @staticmethod
    def _format_shapes(shapes):
        return shapes.replace(',)', ')').replace('((', '(').replace('))', ')')

    def __str__(self):
        str_inputs = OperationShape._format_shapes(str(self.inputs))
        str_outputs = OperationShape._format_shapes(str(self.outputs))
        return f"\n\t\t{{\n\t\t\t{self.attrs},\n\t\t\t{str_inputs}; {str_outputs}\n\t\t}}"

    def __repr__(self):
        str_inputs = OperationShape._format_shapes(str(self.inputs))
        str_outputs = OperationShape._format_shapes(str(self.outputs))
        return f"\n\t\t{{\n\t\t\tattrs: {self.attrs},\n\t\t\tin: {str_inputs}; out: {str_outputs}\n\t\t}}"

    def __eq__(self, other):
        return self.inputs.__eq__(other.inputs) and self.outputs.__eq__(other.outputs)

    def __hash__(self):
        return hash(self.inputs) ^ hash(self.outputs)


def parse_input_shapes(layer):
    input_shapes = []
    start_input = False
    for l in layer:
        if l.name == 'input':
            start_input = True
        if start_input and l.name == 'port':
            dims = l.find_all("dim")
            input_shapes.append(tuple(int(dim.text) for dim in dims))
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
            output_shapes.append(tuple(int(dim.text) for dim in dims))
    return output_shapes


if __name__ == '__main__':
    args = get_arguments()
    with open(args.model, "r") as file:
        content = file.readlines()
        content = "".join(content)
        bs_content = bs(content, "lxml")
        layers = bs_content.find_all("layer")
        layers = sorted(layers, key=lambda l: l.attrs['type'])
        operations = {}
        for layer in layers:
            layer_type = layer.attrs['type']
            layer_version = layer.attrs['version']
            if layer_type not in operations:
                operations[layer_type] = Operation(layer_type, layer_version)
            data_tag = layer.find("data")
            attrs = data_tag.attrs if data_tag else {}
            input_shapes = parse_input_shapes(layer)
            output_shapes = parse_output_shapes(layer)
            operations[layer_type].shapes.add(OperationShape(attrs, input_shapes, output_shapes))

        if args.ops == 'all':
            for op_name, op in operations.items():
                print(f"{op}\n")
        else:
            for op in args.ops:
                print(f"{operations[op]}\n")
