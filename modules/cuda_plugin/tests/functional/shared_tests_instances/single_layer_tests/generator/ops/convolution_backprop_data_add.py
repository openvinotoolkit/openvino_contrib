#!/usr/bin/env python3

from test_definition_base import CommonTestTraits
from test_definition_base import TestParamsProviderBase
import utils


class TestTraits (CommonTestTraits):
    @property
    def operator_ir_type_string(self):
        return 'FusedConvBackpropData'

    @property
    def test_params_provider_class(self):
        return TestParamsProvider

    @property
    def cpp_test_filename(self):
        return 'convolution_backprop_data_add.cpp'

    @property
    def template_filename(self):
        return 'convolution_backprop_data_add.cpp.jinja2'

    @property
    def default_cpp_test_class_name(self):
        return 'ConvolutionBackpropDataAddExtendedLayerTest'


class TestParamsProvider (TestParamsProviderBase):
    def __init__(self, list_of_equal_operators, test_traits):
        super().__init__(list_of_equal_operators, test_traits)

    @property
    def cpp_list_input_shape(self):
        return utils.cpp_list_from_tuple_of_ints(self.op.input_ports[0].shape)

    @property
    def cpp_list_output_shape(self):
        shape_dims_list = self.op.input_ports[2].data_from_connected_const_operator()
        assert len(shape_dims_list) == self.op.input_ports[2].shape[0]
        return utils.cpp_list_from_tuple_of_ints(tuple(shape_dims_list))

    @property
    def cpp_list_kernel(self):
        return utils.cpp_list_from_tuple_of_ints(self.op.input_ports[1].shape[2:])

    @property
    def cpp_list_strides(self):
        return utils.cpp_list_from_tuple_of_ints(self.op.data['strides'].as_tuple_of_int())

    @property
    def cpp_list_pads_begin(self):
        return utils.cpp_list_from_tuple_of_ints(self.op.data['pads_begin'].as_tuple_of_int())

    @property
    def cpp_list_pads_end(self):
        return utils.cpp_list_from_tuple_of_ints(self.op.data['pads_end'].as_tuple_of_int())

    @property
    def cpp_list_dilations(self):
        return utils.cpp_list_from_tuple_of_ints(self.op.data['dilations'].as_tuple_of_int())

    @property
    def cpp_num_output_channels(self):
        return str(self.op.output_ports[0].shape[1])

    @property
    def cpp_auto_pad(self):
        return utils.cpp_ngraph_autopad(self.op.data.get('auto_pad').as_str())
