# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Core
from openvino import convert_model

import pytest
import numpy as np
import os


def run_test(ref_inputs, ref_res, test_onnx=False, threshold=1e-5):
    inputs = {}
    shapes = {}
    for i in range(len(ref_inputs)):
        suffix = '{}'.format(i if i > 0 else '')
        inputs['input' + suffix] = ref_inputs[i]
        shapes['input' + suffix] = ref_inputs[i].shape

    ext_path = os.getenv('CUSTOM_OP_LIB')

    core = Core()
    core.add_extension(ext_path)

    net = core.read_model('model.onnx') if test_onnx else convert_model('model.onnx', extension=ext_path)

    net.reshape(shapes)
    compiled_model = core.compile_model(net, 'CPU')

    out = compiled_model(inputs)
    out = next(iter(out.values()))

    assert ref_res.shape == out.shape
    diff = np.max(np.abs(ref_res - out))
    assert diff <= threshold


@pytest.mark.parametrize("shape", [[5, 120, 2], [4, 240, 320, 2], [3, 16, 240, 320, 2], [4, 5, 16, 31, 2]])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("centered", [False, True])
@pytest.mark.parametrize("test_onnx", [False, True])
@pytest.mark.parametrize("dims", [[1], [1, 2], [2, 3]])
def test_fft(shape, inverse, centered, test_onnx, dims):
    from examples.fft.export_model import export

    if len(shape) == 3 and dims != [1] or \
       len(shape) == 4 and dims in ([1, 2], [2, 3]) or \
       len(shape) == 5 and dims in ([1], [1, 2], [2, 3]) or \
       centered and len(dims) != 2:
        pytest.skip("unsupported configuration")

    if len(shape) == 4 and dims == [1]:
        pytest.skip("Custom FFT executed but there is accuracy error, requires FFT::evaluate fix")


    inp, ref = export(shape, inverse, centered, dims)
    run_test(inp, ref, test_onnx=test_onnx) 


@pytest.mark.parametrize("shape", [[3, 2, 4, 8, 2], [3, 1, 4, 8, 2]])
@pytest.mark.parametrize("test_onnx", [False, True])
def test_complex_mul(shape, test_onnx):
    from examples.complex_mul.export_model import export

    inp, ref = export(other_shape=shape)
    run_test(inp, ref, test_onnx=test_onnx)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("filters", [1, 4])
@pytest.mark.parametrize("kernel_size", [[3, 3, 3], [5, 5, 5], [2, 2, 2]])
@pytest.mark.parametrize("out_pos", [None, 16])
def test_sparse_conv(in_channels, filters, kernel_size, out_pos):
    from examples.sparse_conv.export_model import export

    inp, ref = export(num_inp_points=1000, num_out_points=out_pos, max_grid_extent=4, in_channels=in_channels,
                      filters=filters, kernel_size=kernel_size, transpose=False)
    run_test(inp, ref, test_onnx=True, threshold=1e-4)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("filters", [1, 4])
@pytest.mark.parametrize("kernel_size", [[3, 3, 3], [5, 5, 5]])
@pytest.mark.parametrize("out_pos", [None, 16])
def test_sparse_conv_transpose(in_channels, filters, kernel_size, out_pos):
    from examples.sparse_conv.export_model import export

    inp, ref = export(num_inp_points=1000, num_out_points=out_pos, max_grid_extent=4, in_channels=in_channels,
                      filters=filters, kernel_size=kernel_size, transpose=True)
    run_test(inp, ref, test_onnx=True, threshold=1e-4)


def test_calculate_grid():
    from examples.calculate_grid.export_model import export
    inp, ref = export(num_points=10, max_grid_extent=5)
    run_test(inp, ref, test_onnx=True)
