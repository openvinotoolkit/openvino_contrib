import torch
from collections import namedtuple

from ..hooks import OpenVINOTensor, forward_hook

from mo.utils.error import Error

def p2o(psf, shape):
    pad = []
    for a, b in zip(psf.shape[:-2] + shape, psf.shape):
        pad = [0, a - b] + pad
    otf = torch.nn.functional.pad(psf, pad)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    real, imag = otf[..., 0], otf[..., 1]
    imag = torch.where(torch.abs(imag) < n_ops*2.22e-16, torch.tensor(0).type_as(psf), imag)
    return torch.stack((real, imag), dim=-1)


def deconv_upsample(x, sf=3):
    st = 0
    ch = x.shape[1]
    weights = torch.zeros((ch, 1, sf, sf))
    weights[..., st::sf, st::sf] = 1
    up = torch.nn.functional.conv_transpose2d(x, weights, bias=None,
                                             stride=(sf, sf),
                                             padding=(0, 0),
                                             output_padding=(0, 0),
                                             groups=ch, dilation=(1, 1))
    return up


def cconj(t, inplace=False):
    c = t.clone() if not inplace else t
    return c * torch.tensor([1, -1])


class USRNet(object):
    def __init__(self):
        self.class_name = 'models.network_usrnet.USRNet'

    def hook(self, new_func):
        return lambda *args: new_func(*args)

    def register_hook(self, model):
        import models.network_usrnet
        models.network_usrnet.upsample = deconv_upsample
        models.network_usrnet.p2o = p2o
        models.network_usrnet.cconj = cconj
