# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from .calculate_grid import CalculateGrid


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.calculate_grid = CalculateGrid()

    def forward(self, x):
        return self.calculate_grid.apply(x)


def export(num_points, max_grid_extent):
    # Generate a list of unique positions and add a mantissa
    np.random.seed(32)
    torch.manual_seed(11)

    inp_pos = np.random.randint(0, max_grid_extent, [num_points, 3])
    inp_pos = torch.tensor(inp_pos) + torch.rand(inp_pos.shape, dtype=torch.float32) # [0, 1)

    model = MyModel()
    with torch.no_grad():
        torch.onnx.export(model, (inp_pos), 'model.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(inp_pos).detach().numpy()

    # Pad with sentinel values (-1, 0, 0) and zeros
    ref = np.concatenate((ref, [[-1, 0, 0]]))
    ref = np.pad(ref, ((0, inp_pos.shape[0] - ref.shape[0]), (0, 0)))

    return [inp_pos.detach().numpy()], ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--num_points', type=int, default=10)
    parser.add_argument('--max_grid_extent', type=int, default=5)
    args = parser.parse_args()

    export(args.num_points, args.max_grid_extent)
