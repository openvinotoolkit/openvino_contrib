import torch.nn as nn
import torch

class RotHeadNet(nn.Module):
    def __init__(self, in_channels, num_layers=3, num_filters=256, kernel_size=3, output_kernel_size=1,
                 output_dim=3, freeze=False, with_bias_end=True):
        super(RotHeadNet, self).__init__()

        self.freeze = freeze

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert output_kernel_size == 1 or output_kernel_size == 3, 'Only support kenerl 1 and 3'
        if output_kernel_size == 1:
            pad = 0
        elif output_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))


        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, output_dim, kernel_size=output_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, output_dim, kernel_size=output_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(output_dim))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end and (m.bias is not None):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                return x.detach()
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            return x

