#!/usr/bin/env python3

import os
import argparse
import glob
from jinja2 import Template

from ir_model import IRModel

from cfg_disabled_tests import cfg_disabled_tests
from cfg_overridden_precisions import cfg_overridden_precisions
from cfg_overridden_test_classes import cfg_overridden_test_classes

import ops.convolution as convolution
import ops.mat_mul as mat_mul
import ops.relu as relu
import ops.sigmoid as sigmoid
import ops.clamp as clamp
import ops.swish as swish
import ops.maxpool as maxpool
import ops.avgpool as avgpool
import ops.group_convolution as group_convolution

import ops.convolution_backprop_data_add as convolution_backprop_data_add
import ops.convolution_backprop_data as convolution_backprop_data
import ops.convolution_biasadd_activation as convolution_biasadd_activation

tests_to_generate = [
    convolution.TestTraits(),
    mat_mul.TestTraits(),
    relu.TestTraits(),
    sigmoid.TestTraits(),
    clamp.TestTraits(),
    swish.TestTraits(),
    maxpool.TestTraits(),
    avgpool.TestTraits(),
    group_convolution.TestTraits(),

    # The following test gerators require models with applied graph transformations.
    # If you don't have such models some tests will be dropped (including ones for fused operators).
    # We temporarily disable the following test generators so their test cpp files remain untouched
    # when you don't have the models they require.
    # TODO: Consider to apply graph transformations to original models in this script, generate
    # temporary models with applied transformations and generate corresponding tests (so we don't have
    # to do this manualy).
    #
    # Requires exported models with OpenVINO graph transformations, so the 3rd input tensor (containing
    # output shape parameter) comes from 'Const' operator.
    # convolution_backprop_data.TestTraits(),
    #
    # Requires exported models to contain cuda-fused backward data convolution.
    # convolution_backprop_data_add.TestTraits(),
    #
    # Requires exported models to contain cuda-fused convolutions.
    # convolution_biasadd_activation.TestTraits(),
]


def get_arguments():
    parser = argparse.ArgumentParser(description='Process OpenVINO *.xml model')
    parser.add_argument('--models',
                        required=True,
                        type=str,
                        nargs=1,
                        help='Path to root folder containing IR xml models')
    return parser.parse_args()


def update_cpp_tests_file(filepath, tag_begin, tag_end, generated_content):
    filepath_dst = filepath + ".tmp"
    with open(filepath, "r") as f_src:
        with open(filepath_dst, "w") as f_dst:
            is_autogen_line = False
            for line in f_src:
                if tag_end in line:
                    is_autogen_line = False
                if not is_autogen_line:
                    f_dst.write(line)
                if tag_begin in line:
                    is_autogen_line = True
                    f_dst.write(generated_content)
    os.remove(filepath)
    os.rename(filepath_dst, filepath)


if __name__ == '__main__':
    args = get_arguments()

    generator_directory = os.path.dirname(os.path.abspath(__file__))
    jinja_templates_directory = generator_directory + '/ops'
    cpp_testfile_directory = os.path.dirname(generator_directory)

    models = list()
    model_files = glob.glob(args.models[0] + '/**/*.xml', recursive=True)
    model_files.sort()
    for filepath in model_files:
        print("Loading IR: {} ... ".format(filepath), end='')
        models.append(IRModel(filepath))
        print("done.")

    print("Merging duplicate operators...")

    # Determine duplicate operators
    grouped_duplicate_operators = dict()  # 'op1' -> list of equal operators [op1, op2, ...]
    for model in models:
        for op in model.operators:
            list_of_equal_operators = grouped_duplicate_operators.get(op)
            if not list_of_equal_operators:
                list_of_equal_operators = list()
                grouped_duplicate_operators[op] = list_of_equal_operators
            list_of_equal_operators.append(op)

    # Map IR operator type strings into corresponding test traits descriptors
    test_traits_dict = dict()
    for test_traits in tests_to_generate:
        test_traits_dict[test_traits.operator_ir_type_string] = test_traits

    # Wrap each list of duplicates with test parameter provider object which is used as a data
    # source during test generation. One provider represents one generated test.
    test_params_providers = dict()  # 'operator IR type str' -> [test_data_provider1, test_data_provider2, ...]
    for list_of_equal_operators in grouped_duplicate_operators.values():
        op_ir_type_str = list_of_equal_operators[0].type
        list_of_providers = test_params_providers.get(op_ir_type_str)
        if not list_of_providers:
            list_of_providers = list()
            test_params_providers[op_ir_type_str] = list_of_providers
        test_traits = test_traits_dict.get(op_ir_type_str)
        if test_traits:
            list_of_providers.append(test_traits.test_params_provider_class(list_of_equal_operators, test_traits))

    # Uncomment the following line to dump param providers you are implementing (includes op attributes and tensor
    # shapes)
    # print(test_params_providers['FusedConvolution'])

    # Map test alias strings into test params providers (so we can apply configs, like 'disabled' etc.)
    aliased_providers = dict()
    for providers in test_params_providers.values():
        for provider in providers:
            for alias_str in provider.aliases:
                aliased_providers[alias_str] = provider

    # Apply disabled tests config
    for alias in cfg_disabled_tests:
        provider = aliased_providers.get(alias)
        if provider:
            provider.cfg_test_is_disabled = True
        else:
            print("WARNING: failed to disable test using test alias '{}'".format(alias))

    # Override tested net precisions
    for alias, net_precisions_list in cfg_overridden_precisions.items():
        provider = aliased_providers.get(alias)
        if provider:
            provider.cfg_net_precisions_list = net_precisions_list
        else:
            print("WARNING: failed to override test net precisions using test alias '{}'".format(alias))

    # Override default C++ test classes
    for alias, cpp_class_name in cfg_overridden_test_classes.items():
        provider = aliased_providers.get(alias)
        if provider:
            provider.cfg_cpp_test_class_name = cpp_class_name
        else:
            print("WARNING: failed to override C++ test class using test alias '{}'".format(alias))

    # Generate tests
    for test_traits in tests_to_generate:
        test_params_providers_list = test_params_providers.pop(test_traits.operator_ir_type_string, list())
        print("Generating tests for '{}' ({} tests/precision)...".format(test_traits.operator_ir_type_string,
                                                                         str(len(test_params_providers_list))))
        test_params_providers_list.sort(key=lambda p: p.op.test_identity)
        template = Template(open(jinja_templates_directory + '/' + test_traits.template_filename).read())
        generated_tests_string = template.render(operators=test_params_providers_list, test_traits=test_traits)
        update_cpp_tests_file(cpp_testfile_directory + '/' + test_traits.cpp_test_filename,
                              test_traits.cpp_test_file_begin_tag,
                              test_traits.cpp_test_file_end_tag,
                              generated_tests_string)

    print('DONE.')

    no_ops = {
        'Parameter',
        'Result',
        'Const',
        'Squeeze',
        'Unsqueeze',
        'Reshape',
    }
    for nop in no_ops:
        test_params_providers.pop(nop)
    print('\nThe rest of operators:\n\t{}'.format('\n\t'.join(test_params_providers.keys())))
