import torch
import torch.nn as nn

from transformer import GeometricTransformer
from model_utils import (
    compute_feature_similarity,
    aug_pose_noise,
    compute_coarse_Rt,
)
from loss_utils import compute_correspondence_loss



class CoarsePointMatching(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(CoarsePointMatching, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock
        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)
        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * .02)
        self.transformers = nn.ModuleList([
            GeometricTransformer(
                blocks=['self', 'cross'],
                d_model = cfg.hidden_dim,
                num_heads = 4,
                dropout=None,
                activation_fn='ReLU',
                return_attention_scores=False,
            ) for _ in range(self.nblock)
        ])

    def forward(self, p1, f1, geo1, p2, f2, geo2, radius, model):
        B = f1.size(0)

        f1 = self.in_proj(f1)
        f1 = torch.cat([self.bg_token.repeat(B,1,1), f1], dim=1) # adding bg
        f2 = self.in_proj(f2)
        f2 = torch.cat([self.bg_token.repeat(B,1,1), f2], dim=1) # adding bg

        atten_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, f2, geo2)

            if self.training or idx==self.nblock-1:
                atten_list.append(compute_feature_similarity(
                    self.out_proj(f1),
                    self.out_proj(f2),
                    self.cfg.sim_type,
                    self.cfg.temp,
                    self.cfg.normalize_feat
                ))
                
        # 只返回粗匹配的初始R/t
        # 训练时可返回更多
        init_R, init_t = compute_coarse_Rt(
            atten_list[-1], p1, p2,
            model / (radius.reshape(-1, 1, 1) + 1e-6),
            self.cfg.nproposal1, self.cfg.nproposal2,
        )
        return init_R, init_t
