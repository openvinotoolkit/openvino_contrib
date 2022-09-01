#!/usr/bin/env python3

import ir_model

#
# The same operator can be defined using different sets of attributes
# because some attributes have default values. The purpose of this code
# is to use equal python sets of attributes for the same operators, so we
# can detect duplicates and avoid testing the same shapes more than once.
#


#
# See 'openvino/docs/ops/convolution/Convolution_1.md'
#
def preprocess_ir_op_Convolution(op):
    _attribute_set_default_if_none(op, 'auto_pad', 'explicit')
    # Fix: There should be no 'output_padding' attribute in 'Convolution' nodes
    op.data.pop('output_padding', None)
    _reset_padding_attrs_if_not_explicit(op, len(op.input_ports[0].shape) - 2)


#
# See '/mnt/data/space/openvino/docs/ops/convolution/ConvolutionBackpropData_1.md'
#
def preprocess_ir_op_ConvolutionBackpropData(op):
    num_spatial_dims = len(op.input_ports[0].shape) - 2
    _attribute_set_default_if_none(op, 'auto_pad', 'explicit')
    _reset_padding_attrs_if_not_explicit(op, num_spatial_dims)
    _attribute_set_default_if_none(op, 'output_padding',
                                   _csv_int_attribute_zeros(num_spatial_dims))


#
# See 'openvino/docs/ops/pooling/AvgPool_1.md'
#
def preprocess_ir_op_AvgPool(op):
    _attribute_set_default_if_none(op, 'auto_pad', 'explicit')
    _attribute_set_default_if_none(op, 'rounding_type', 'floor')
    _reset_padding_attrs_if_not_explicit(op, len(op.input_ports[0].shape) - 2)
    _fix_boolean_attr(op, 'exclude-pad')


#
# See 'openvino/docs/ops/pooling/MaxPool_1.md'
#
def preprocess_ir_op_MaxPool(op):
    _attribute_set_default_if_none(op, 'auto_pad', 'explicit')
    _attribute_set_default_if_none(op, 'rounding_type', 'floor')
    _reset_padding_attrs_if_not_explicit(op, len(op.input_ports[0].shape) - 2)


#
# See 'openvino/docs/ops/matrix/MatMul_1.md'
#
def preprocess_ir_op_MatMul(op):
    _fix_boolean_attr(op, 'transpose_a')
    _fix_boolean_attr(op, 'transpose_b')


#
# Common
#
def _attribute_set_default_if_none(op, attr_name, default_value):
    op.data[attr_name] = op.data.get(attr_name) or ir_model.AttrValue(default_value)


def _reset_padding_attrs_if_not_explicit(op, num_spatial_dims):
    if not op.data['auto_pad'].as_str() == 'explicit':
        _csv_int_attribute_set_zeros(op, 'pads_begin', num_spatial_dims)
        _csv_int_attribute_set_zeros(op, 'pads_end', num_spatial_dims)


def _csv_int_attribute_set_zeros(op, attr_name, size):
    op.data[attr_name] = ir_model.AttrValue(_csv_int_attribute_zeros(size))


def _csv_int_attribute_zeros(size):
    return ','.join(['0'] * size)


def _fix_boolean_attr(op, attr_name):
    attr_str = op.data[attr_name].as_str()
    attr_str = {'1': 'true', '0': 'false'}.get(attr_str) or attr_str
    op.data[attr_name] = ir_model.AttrValue(attr_str)
