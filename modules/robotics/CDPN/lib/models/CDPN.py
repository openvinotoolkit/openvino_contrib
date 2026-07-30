import os
from easydict import EasyDict as edict

import torch
import torch.nn as nn

class CDPN(nn.Module):
    def __init__(self, backbone, rot_head_net, trans_head_net):
        super(CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net

    def forward(self, x):                     # x.shape [bs, 3, 256, 256]
        features = self.backbone(x)           # features.shape [bs, 2048, 8, 8]
        cc_maps = self.rot_head_net(features) # joints.shape [bs, 1152, 64, 64]
        trans = self.trans_head_net(features)
        return cc_maps, trans
