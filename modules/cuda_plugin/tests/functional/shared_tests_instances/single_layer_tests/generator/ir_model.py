#!/usr/bin/env python3

import os
import glob
import struct
from bs4 import BeautifulSoup
import ir_op_preprocessing


class IRModelID:
    def __init__(self, name, precision):
        assert type(name) == str
        assert type(precision) == str
        self.name = name
        self.precision = precision

    def __repr__(self):
        return '{}:{}'.format(self.name, self.precision)

    def __eq__(self, other):
        return self.name.__eq__(other.name) and self.precision.__eq__(other.precision)

    def __hash__(self):
        return hash(self.name) ^ hash(self.precision)


class IRModel:
    def __init__(self, ir_xml_filepath):
        self.id = self.__model_id(ir_xml_filepath)
        self.bin_filepath = self.__bin_filepath(ir_xml_filepath)
        self.operators = list()
        bs_content = self.__load_xml(ir_xml_filepath)
        for xml_element_layer in bs_content.find('layers').find_all('layer', recursive=False):
            self.operators.append(IROperator(self, xml_element_layer))
        self._connect_operators(bs_content)

    @staticmethod
    def __bin_filepath(ir_xml_filepath):
        return os.path.splitext(ir_xml_filepath)[0] + '.bin'

    @staticmethod
    def __load_xml(ir_xml_filepath):
        with open(ir_xml_filepath, 'r') as file:
            content = "".join(file.readlines())
            return BeautifulSoup(content, 'xml')

    @staticmethod
    def __model_id(ir_xml_filepath):
        parent_directory = os.path.dirname(ir_xml_filepath)
        model_precision = os.path.basename(parent_directory)
        model_name_directory = os.path.dirname(parent_directory)
        model_name = os.path.basename(model_name_directory)
        has_sibling_models = len(glob.glob(parent_directory + '/*.xml')) > 1
        if (has_sibling_models):
            xml_filename = os.path.splitext(os.path.basename(ir_xml_filepath))[0]
            model_name += '-' + xml_filename
        return IRModelID(model_name, model_precision)

    def _connect_operators(self, bs_content):
        ports = dict()  # Port.ID -> Port
        for op in self.operators:
            for port in op.input_ports:
                ports[port.id] = port
            for port in op.output_ports:
                ports[port.id] = port
        for xml_element_edge in bs_content.find('edges').find_all('edge'):
            from_port_id = Port.ID(IROperatorID(self.id, xml_element_edge.attrs['from-layer']),
                                   xml_element_edge.attrs['from-port'])
            to_port_id = Port.ID(IROperatorID(self.id, xml_element_edge.attrs['to-layer']),
                                 xml_element_edge.attrs['to-port'])
            from_port = ports.get(from_port_id)
            to_port = ports.get(to_port_id)
            if from_port and to_port:
                from_port.connected_port = to_port
                to_port.connected_port = from_port
                # print('\tConnect {} to {}'.format(from_port, to_port))
            else:
                # print('\tWARNING: Failed to connect port {} to {}'.format(from_port, to_port))
                pass

    def read_bin_file(self, offset, bsize) -> bytes:
        with open(self.bin_filepath, 'rb') as file:
            file.seek(offset)
            return file.read(bsize)


class IROperatorID:
    def __init__(self, modelID, ir_layer_id):
        assert type(modelID) == IRModelID
        assert type(ir_layer_id) == str
        self.modelID = modelID
        self.layer_id = ir_layer_id

    def alias_str(self):
        return '{}:opid{}'.format(self.modelID.name, self.layer_id)

    def __repr__(self):
        return '{}:opid{}'.format(str(self.modelID), self.layer_id)

    def __eq__(self, other):
        return self.modelID.__eq__(other.modelID) and self.layer_id.__eq__(other.layer_id)

    def __hash__(self):
        return hash(self.modelID) ^ hash(self.layer_id)


class IROperator:
    def __init__(self, model, xml_element_layer):
        self.model = model
        self.id = IROperatorID(model.id, xml_element_layer.attrs['id'])
        self.type = xml_element_layer.attrs['type']
        self.version = xml_element_layer.attrs['version']

        self.data = dict()
        xml_element_data = xml_element_layer.find('data')
        xml_element_data_attrs = xml_element_data.attrs if xml_element_data else dict()
        for key_str, value_str in xml_element_data_attrs.items():
            self.data[key_str] = AttrValue(value_str)

        self.input_ports = []
        xml_element_input = xml_element_layer.find('input')
        for xml_element_port in xml_element_input.find_all('port') if xml_element_input else []:
            self.input_ports.append(Port(self, xml_element_port))
        self.input_ports = tuple(self.input_ports)

        self.output_ports = []
        xml_element_ouput = xml_element_layer.find('output')
        for xml_element_port in xml_element_ouput.find_all('port') if xml_element_ouput else []:
            self.output_ports.append(Port(self, xml_element_port))
        self.output_ports = tuple(self.output_ports)

        try:
            preprocessor_func = getattr(ir_op_preprocessing, 'preprocess_ir_op_' + self.type)
            preprocessor_func(self)
        except AttributeError:
            pass

    @property
    def test_identity(self) -> str:
        return 'IROperator {}:{}, Attrs: [{}], In: [{}], Out: [{}]'.format(
            self.version,
            self.type,
            ','.join(map(lambda t: "'{}': {}".format(t[0], t[1].test_identity), sorted(self.data.items()))),
            ','.join(map(lambda p: p.test_identity, self.input_ports)),
            ','.join(map(lambda p: p.test_identity, self.output_ports)))

    def attributes_as_str(self):
        items = []
        for key, attr_value in sorted(self.data.items()):
            items.append("'{}': '{}'".format(key, attr_value.as_str()))
        return '{' + ", ".join(items) + '}'

    def inputs_as_str(self):
        return ", ".join(map(lambda p: p.as_str(), self.input_ports))

    def outputs_as_str(self):
        return ", ".join(map(lambda p: p.as_str(), self.output_ports))

    def const_data(self):
        if not self.type == 'Const':
            raise TypeError("IROperator: 'const_data()' is not supported by '{}'".format(self.type))
        bs = self.model.read_bin_file(self.data['offset'].as_int(), self.data['size'].as_int())
        el_type = self.data['element_type'].as_str()
        formats = {'f16': '<e', 'f32': '<f', 'i64': '<q', 'i32': '<i'}
        return list(map(lambda t: t[0], struct.iter_unpack(formats[el_type], bs)))

    def __repr__(self):
        return "\nIROperator({}::{}, id={}, \n\t data={}, \n\t inputs={}, \n\t outputs={})".format(
            self.version, self.type, self.id, self.data, self.input_ports, self.output_ports)

    def __eq__(self, other):
        return self.test_identity.__eq__(other.test_identity)

    def __hash__(self):
        return hash(self.test_identity)


class Port:
    def __init__(self, operator, xml_element_port):
        assert type(operator) == IROperator
        self.host_operator = operator
        self.id = self.ID(operator.id, xml_element_port.attrs['id'])
        self.precision = xml_element_port.attrs.get('precision')
        dims = xml_element_port.find_all('dim')
        if len(dims) > 0:
            self.shape = tuple(int(dim.text) for dim in dims)
        else:
            self.shape = tuple([1])
        self.connected_port = None

    @property
    def test_identity(self) -> str:
        return "Port(shape={})".format(self.shape)

    def as_str(self) -> str:
        return "({})".format(", ".join(map(lambda i: str(i), self.shape)))

    def data_from_connected_const_operator(self) -> IROperator:
        return self.connected_port.host_operator.const_data()

    class ID:
        def __init__(self, operatorID, ir_port):
            assert type(operatorID) == IROperatorID
            assert type(ir_port) == str
            self.operatorID = operatorID
            self.ir_port = ir_port

        def __repr__(self):
            return '{}:port{}'.format(str(self.operatorID), self.ir_port)

        def __eq__(self, other):
            return self.operatorID.__eq__(other.operatorID) and self.ir_port.__eq__(other.ir_port)

        def __hash__(self):
            return hash(self.operatorID) ^ hash(self.ir_port)

    def __repr__(self):
        return "Port({}, shape={})".format(self.id, self.shape)

    def __eq__(self, other):
        return self.test_identity.__eq__(other.test_identity)

    def __hash__(self):
        return hash(self.test_identity)


class AttrValue:
    def __init__(self, value_str):
        # Drop all whitespaces and convert to lowercase
        self._value_str = ''.join(value_str.split()).lower()

    @property
    def test_identity(self) -> str:
        try:
            return "AttrValue({})".format(','.join(map(lambda i: str(i), self.as_tuple_of_int())))
        except ValueError:
            return "AttrValue('{}')".format(self.as_str())

    def as_str(self):
        return self._value_str

    def as_int(self):
        return int(self.as_str())

    def as_float(self):
        return float(self.as_str())

    def as_tuple_of_str(self):
        return tuple(map(lambda s: s.strip(), self.as_str().split(',')))

    def as_tuple_of_int(self):
        s = self.as_str().strip()
        return tuple(map(lambda s: int(s), self.as_tuple_of_str())) if len(s) else tuple()

    def __repr__(self):
        return "AttrValue({})".format(self.as_str())

    def __str__(self):
        return "'{}'".format(self.as_str())

    def __eq__(self, other):
        return self.test_identity.__eq__(other.test_identity)

    def __hash__(self):
        return hash(self.test_identity)
