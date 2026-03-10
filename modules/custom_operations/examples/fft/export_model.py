# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .fft import FFT


class MyModel(nn.Module):
    def __init__(self, inverse, centered, dims):
        super(MyModel, self).__init__()
        self.inverse = inverse
        self.centered = centered
        self.dims = dims
        self.fft = FFT()

    def forward(self, x):
        return self.fft.apply(x, self.inverse, self.centered, self.dims)


def export(shape, inverse, centered, dims):
    np.random.seed(324)
    torch.manual_seed(32)

    model = MyModel(inverse, centered, dims)
    inp = Variable(torch.randn(shape))
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, inp, 'model.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

    ref = model(inp)
    return [inp.detach().numpy()], ref.detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--shape', type=int, nargs='+', default=[5, 3, 6, 8, 2])
    parser.add_argument('--inverse', action='store_true')
    parser.add_argument('--centered', action='store_true')
    parser.add_argument('--dims', type=int, nargs='+', default=[2, 3])
    args = parser.parse_args()
    export(args.shape, args.inverse, args.centered, args.dims)
