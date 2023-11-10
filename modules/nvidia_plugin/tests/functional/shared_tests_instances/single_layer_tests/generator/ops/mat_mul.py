#!/usr/bin/env python3

from test_definition_base import CommonTestTraits
from test_definition_base import TestParamsProviderBase
import utils


class TestTraits (CommonTestTraits):
    @property
    def operator_ir_type_string(self):
        return 'MatMul'

    @property
    def test_params_provider_class(self):
        return TestParamsProvider

    @property
    def cpp_test_filename(self):
        return 'mat_mul.cpp'

    @property
    def template_filename(self):
        return 'mat_mul.cpp.jinja2'

    @property
    def default_cpp_test_class_name(self):
        return 'MatMulLayerTest'


class TestParamsProvider (TestParamsProviderBase):
    def __init__(self, list_of_equal_operators, test_traits):
        super().__init__(list_of_equal_operators, test_traits)

    @property
    def cpp_shape_related_params(self):
        return 'ShapeRelatedParams{{ {{ {}, {} }}, {{ {}, {} }} }}'.format(
            utils.cpp_list_from_tuple_of_ints(self.op.input_ports[0].shape),
            utils.cpp_bool(self.op.data['transpose_a'].as_str()),
            utils.cpp_list_from_tuple_of_ints(self.op.input_ports[1].shape),
            utils.cpp_bool(self.op.data['transpose_b'].as_str())
        )

    @property
    def cpp_additional_config(self):
        return 'std::map<std::string, std::string> {}'

    @property
    def cpp_secondary_input_types(self):
        return ('std::vector<InputLayerType> {'
                'InputLayerType::CONSTANT, '
                'InputLayerType::PARAMETER'
                '}')
