#!/usr/bin/env python3

"""
 Copyright (C) 2018-2022 Intel Corporation

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

import os
import sys
import logging as log

import openvino.tools.mo as mo
from openvino.tools.mo.main import main, print_argv
from openvino.tools.mo.utils.cli_parser import get_common_cli_parser

from openvino.tools.mo.utils.cli_parser import get_placeholder_shapes, parse_tuple_pairs, \
    get_mean_scale_dictionary, get_freeze_placeholder_values
from openvino.tools.mo.front.common.replacement import FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.utils import import_extensions
from openvino.tools.mo.pipeline.unified import unified_pipeline


def get_front_classes():
    front_classes = [FrontExtractorOp, FrontReplacementOp, FrontReplacementPattern, FrontReplacementSubgraph]
    return front_classes

# A copy of mo.main.prepare_ir but adopted for PyTorch conversion
def _prepare_ir(argv, old_api=False):
    log.debug(str(argv))
    log.debug("Model Optimizer started")

    model_name = "<UNKNOWN_NAME>"
    if argv.model_name:
        model_name = argv.model_name
    elif argv.input_model:
        model_name = argv.input_model.__class__.__name__
    argv.model_name = model_name

    log.debug('Output model name would be {}{{.xml, .bin}}'.format(argv.model_name))

    if not argv.silent:
        print_argv(argv, False, False, False, False, False, argv.model_name)

    if argv.scale and argv.scale_values:
        raise Error(
            'Both --scale and --scale_values are defined. Specify either scale factor or scale values per input ' +
            'channels. ' + refer_to_faq_msg(19))

    if argv.scale and argv.scale < 1.0:
        log.error("The scale value is less than 1.0. This is most probably an issue because the scale value specifies "
                  "floating point value which all input values will be *divided*.", extra={'is_warning': True})

    argv.output = argv.output.split(',') if argv.output else None

    argv.inputs_list, argv.placeholder_shapes, argv.placeholder_data_types = get_placeholder_shapes(argv.input, argv.input_shape, argv.batch)

    mean_values = parse_tuple_pairs(argv.mean_values)
    scale_values = parse_tuple_pairs(argv.scale_values)
    mean_scale = get_mean_scale_dictionary(mean_values, scale_values, argv.input)
    argv.mean_scale_values = mean_scale

    if not os.path.exists(argv.output_dir):
        try:
            os.makedirs(argv.output_dir)
        except PermissionError as e:
            raise Error("Failed to create directory {}. Permission denied! " +
                        refer_to_faq_msg(22),
                        argv.output_dir) from e
    else:
        if not os.access(argv.output_dir, os.W_OK):
            raise Error("Output directory {} is not writable for current user. " +
                        refer_to_faq_msg(22), argv.output_dir)

    log.debug("Placeholder shapes : {}".format(argv.placeholder_shapes))

    ret_res = 1
    if hasattr(argv, 'extensions') and argv.extensions and argv.extensions != '':
        extensions = argv.extensions.split(',')
    else:
        extensions = None

    argv.freeze_placeholder_with_value, argv.input = get_freeze_placeholder_values(argv.input,
                                                                                   argv.freeze_placeholder_with_value)

    import_extensions.load_dirs(argv.framework, extensions, get_front_classes)

    graph = unified_pipeline(argv)
    if old_api:
        return graph
    else:
        return graph, None


def convert(model, **args):
    mo.main.prepare_ir = _prepare_ir

    parser = get_common_cli_parser()
    parser.set_defaults(input_model=model,
                        extensions=os.path.join(os.path.dirname(__file__), 'mo_extensions'),
                        ie_is_available=False)
    for arg, value in args.items():
        parser.set_defaults(**{arg: str(value)})
    parser.set_defaults(is_dynamic=args.get("is_dynamic", True))

    # Replace original parser to ignore global sys.argv
    origin_parse = parser.parse_args
    parser.parse_args = lambda: origin_parse([])

    err = None
    try:
        err = main(parser, None, 'pytorch')
    except:
        if err is None:
            mo.main.prepare_ir = lambda argv : _prepare_ir(argv, old_api=True)
            err = main(parser, 'pytorch')
    if err:
        raise Exception('model conversion failed')
