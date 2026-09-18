#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
SAM 3D Objects — OpenVINO Export Script (Standalone)

Fully standalone — no sam3d_objects imports.
All architectures re-implemented inline; weights loaded from checkpoints.
"""
from __future__ import annotations
import argparse, json, math, numpy as np, os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import openvino as ov
except ImportError:
    raise ImportError("OpenVINO Python API not found")

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ===========================================================================
# Architecture Constants
# ===========================================================================
DINO_INPUT_SIZE = 518
DINO_EMBED_DIM = 1024
DINO_NUM_TOKENS = 1374  # CLS + 4 reg + 37x37 patches
POINTPATCH_INPUT_SIZE = 256
POINTPATCH_PATCH_SIZE = 8
POINTPATCH_NUM_WINDOWS = (256 // 8) ** 2  # 1024
POINTPATCH_EMBED_DIM = 512
COND_EMBED_DIM = 1024
SS_COND_NUM_TOKENS = 4 * DINO_NUM_TOKENS + 2 * POINTPATCH_NUM_WINDOWS  # 7544
SLAT_COND_NUM_TOKENS = 4 * DINO_NUM_TOKENS   # 5496
SLAT_COND_CHANNELS = COND_EMBED_DIM  # 1024
SS_MODEL_CHANNELS = 1024
SS_NUM_BLOCKS = 24
SS_NUM_HEADS = 16
SLAT_MODEL_CHANNELS = 1024
SLAT_NUM_BLOCKS = 24
SLAT_NUM_HEADS = 16
SS_LATENT_NAMES = ["shape", "6drotation_normalized", "translation", "scale", "translation_scale"]
SS_LATENT_IN_CHANNELS = {"shape": 8, "6drotation_normalized": 6, "translation": 3, "scale": 3, "translation_scale": 1}
SS_LATENT_NUM_TOKENS = {"shape": 4096, "6drotation_normalized": 1, "translation": 1, "scale": 1, "translation_scale": 1}
SS_SHAPE_LATENT_TOKENS = 4096
SS_POSE_MERGED_TOKENS = 4  # merged pose latent tokens
# Decoder constants (different from generator: 768 channels, 12 blocks)
SLAT_DEC_MODEL_CHANNELS = 768
SLAT_DEC_NUM_HEADS = 12
SLAT_DEC_NUM_BLOCKS = 12
SLAT_DEC_LATENT_CHANNELS = 8
SLAT_DEC_GS_OUT_CHANNELS = 448    # 32 Gaussians × (3+3+3+4+1)
POINTPATCH_OUTER_DIM = 512        # point_proj output dim

# ===========================================================================
# Checkpoint helpers
# ===========================================================================
def load_checkpoint(path: Path) -> dict:
    s = str(path)
    if s.endswith(".safetensors"):
        if not HAS_SAFETENSORS:
            raise ImportError("safetensors not installed")
        return load_safetensors(s, device="cpu")
    sd = torch.load(s, map_location="cpu", weights_only=False)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd

def filter_prefix(sd: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

def save_ov(model: ov.Model, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{name}.xml"
    ov.save_model(model, str(p), compress_to_fp16=True)
    print(f"  ✓ {name}.xml")
    return p

def torch_to_ov(module: nn.Module, example, outdir: Path, name: str) -> Path:
    """Convert a PyTorch module to OpenVINO IR using direct OV conversion.

    Uses ov.convert_model() directly on the PyTorch nn.Module, which uses
    torch.jit.trace internally and is much faster than the ONNX exporter
    for large models with many repeated blocks.
    """
    module.eval()
    outdir.mkdir(parents=True, exist_ok=True)
    # Normalise example to list for ov.convert_model
    if isinstance(example, torch.Tensor):
        example_input = [example]
    else:
        example_input = list(example)
    with torch.no_grad():
        ov_m = ov.convert_model(module, example_input=example_input)
    return save_ov(ov_m, outdir, name)

# ===========================================================================
# Layer primitives
# ===========================================================================
class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)

class ChannelLayerNorm32(nn.LayerNorm):
    """LayerNorm applied along channel dim of (B, C, D, H, W)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        DIM = x.dim()
        x = x.permute(0, *range(2, DIM), 1).contiguous()
        x = super().forward(x.float()).to(x.dtype)
        x = x.permute(0, DIM - 1, *range(1, DIM - 1)).contiguous()
        return x

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))
    def forward(self, x):
        return (F.normalize(x, dim=-1) * self.gamma * self.scale).to(x.dtype)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int, freq_size: int = 256):
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, hidden, bias=True),
        )
    @staticmethod
    def sincos(t: torch.Tensor, dim: int, max_p: int = 10000) -> torch.Tensor:
        half = dim // 2
        f = torch.exp(-math.log(max_p) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * f[None]
        e = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            e = torch.cat([e, torch.zeros_like(e[:, :1])], dim=-1)
        return e
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return self.mlp(self.sincos(t, self.freq_size).to(self.mlp[0].weight.dtype))

# ===========================================================================
# DINOv2 Wrapper
# ===========================================================================
class DinoWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, input_size: int = 518,
                 prenorm: bool = False):
        super().__init__()
        self.backbone = backbone
        self.sz = input_size
        self.prenorm = prenorm
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (self.sz, self.sz), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        out = self.backbone.forward_features(x)
        if self.prenorm:
            feat = out["x_prenorm"]
            return F.layer_norm(feat, feat.shape[-1:])
        return torch.cat([out["x_norm_clstoken"].unsqueeze(1), out["x_norm_patchtokens"]], dim=1)

def load_dino_backbone(full_sd: dict, cond_prefix: str, idx: int) -> nn.Module:
    # NOTE: DINOv2 is loaded from torch.hub GitHub HEAD (unpinned), matching the
    # SAM3D baseline. The checkpoint's DINO weights are identical to the raw hub
    # pretrained weights (no fine-tuning), so this export is a faithful,
    # deterministic reproduction of the checkpoint.  Note that different dinov2
    # hub commits can produce slightly different backbone outputs.
    import warnings
    print(f"    Loading DINOv2 (module_list.{idx}.backbone) via torch.hub...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg",
                            source="github", verbose=False)
    dino_sd = filter_prefix(full_sd, cond_prefix + f"module_list.{idx}.backbone.")
    # Use strict=True so checkpoint weights fully determine the model — no hub defaults leak.
    # The checkpoint contains all backbone parameters.
    missing, unexpected = bb.load_state_dict(dino_sd, strict=True)
    if missing:
        print(f"    WARNING: {len(missing)} missing keys in DINO backbone (using hub defaults): {missing[:3]}")
    bb.eval().requires_grad_(False)
    return bb

# ===========================================================================
# ProjectionNet (SwiGLU FeedForward = Llama3)
# ===========================================================================
class ProjectionNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(self.norm(x))) * self.w3(self.norm(x)))
    def load_sd(self, sd: dict, prefix: str):
        self.norm.weight.data.copy_(sd[prefix + "0.weight"])
        self.norm.bias.data.copy_(sd[prefix + "0.bias"])
        self.w1.weight.data.copy_(sd[prefix + "1.w1.weight"])
        self.w2.weight.data.copy_(sd[prefix + "1.w2.weight"])
        self.w3.weight.data.copy_(sd[prefix + "1.w3.weight"])

# ===========================================================================
# SS Backbone: MOT block
# ===========================================================================
class SSMOTBlock(nn.Module):
    """MOTModulatedTransformerCrossBlock for SS (2 latent streams: shape, pose)."""
    def __init__(self, ch: int, ctx_ch: int, heads: int):
        super().__init__()
        self.ch = ch; self.H = heads; self.hd = ch // heads
        self.norm2_s = LayerNorm32(ch, elementwise_affine=True, eps=1e-6)
        self.norm2_p = LayerNorm32(ch, elementwise_affine=True, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(ch, 6*ch, bias=True))
        # Self-attn per-latent projections
        self.sa_qkv_s = nn.Linear(ch, ch*3, bias=True)
        self.sa_qkv_p = nn.Linear(ch, ch*3, bias=True)
        self.sa_qrms_s = MultiHeadRMSNorm(self.hd, heads)
        self.sa_krms_s = MultiHeadRMSNorm(self.hd, heads)
        self.sa_qrms_p = MultiHeadRMSNorm(self.hd, heads)
        self.sa_krms_p = MultiHeadRMSNorm(self.hd, heads)
        self.sa_out_s = nn.Linear(ch, ch, bias=True)
        self.sa_out_p = nn.Linear(ch, ch, bias=True)
        # Cross-attn per-latent
        self.ca_q_s = nn.Linear(ch, ch, bias=True)
        self.ca_kv_s = nn.Linear(ctx_ch, ch*2, bias=True)
        self.ca_o_s = nn.Linear(ch, ch, bias=True)
        self.ca_q_p = nn.Linear(ch, ch, bias=True)
        self.ca_kv_p = nn.Linear(ctx_ch, ch*2, bias=True)
        self.ca_o_p = nn.Linear(ch, ch, bias=True)
        # MLP per-latent
        mlp_h = ch * 4
        self.mlp_s = nn.Sequential(nn.Linear(ch, mlp_h), nn.GELU(approximate="tanh"), nn.Linear(mlp_h, ch))
        self.mlp_p = nn.Sequential(nn.Linear(ch, mlp_h), nn.GELU(approximate="tanh"), nn.Linear(mlp_h, ch))

    def _ln(self, x):
        return F.layer_norm(x.float(), [self.ch]).to(x.dtype)

    def _sa(self, s, p):
        B, Ls, C = s.shape
        _, Lp, _ = p.shape
        qkv_s = self.sa_qkv_s(s).reshape(B,Ls,3,self.H,self.hd)
        qkv_p = self.sa_qkv_p(p).reshape(B,Lp,3,self.H,self.hd)
        qs, ks, vs = qkv_s.unbind(2)
        qp, kp, vp = qkv_p.unbind(2)
        qs = self.sa_qrms_s(qs); ks = self.sa_krms_s(ks)
        qp = self.sa_qrms_p(qp); kp = self.sa_krms_p(kp)
        # Shape attends only itself
        qs=qs.permute(0,2,1,3); ks=ks.permute(0,2,1,3); vs=vs.permute(0,2,1,3)
        hs = F.scaled_dot_product_attention(qs,ks,vs).permute(0,2,1,3).reshape(B,Ls,C)
        hs = self.sa_out_s(hs)
        # Pose attends pose + shape(detached)
        qp=qp.permute(0,2,1,3)
        kp_all = torch.cat([kp.permute(0,2,1,3), ks.detach()], dim=2)
        vp_all = torch.cat([vp.permute(0,2,1,3), vs.detach()], dim=2)
        hp = F.scaled_dot_product_attention(qp,kp_all,vp_all).permute(0,2,1,3).reshape(B,Lp,C)
        hp = self.sa_out_p(hp)
        return hs, hp

    def _ca(self, x, ctx, q_l, kv_l, o_l):
        B,L,C = x.shape; Lk=ctx.shape[1]
        q=q_l(x).reshape(B,L,self.H,self.hd).permute(0,2,1,3)
        kv=kv_l(ctx).reshape(B,Lk,2,self.H,self.hd)
        k,v=kv.unbind(2); k=k.permute(0,2,1,3); v=v.permute(0,2,1,3)
        h=F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(B,L,C)
        return o_l(h)

    def forward(self, shape, pose, t_emb, cond):
        sh_m,sc_m,g_m,sh_f,sc_f,g_f = self.adaLN(t_emb).chunk(6,dim=1)
        # SA
        ns=self._ln(shape)*(1+sc_m.unsqueeze(1))+sh_m.unsqueeze(1)
        np_=self._ln(pose)*(1+sc_m.unsqueeze(1))+sh_m.unsqueeze(1)
        hs,hp=self._sa(ns,np_)
        shape=shape+hs*g_m.unsqueeze(1); pose=pose+hp*g_m.unsqueeze(1)
        # CA
        shape=shape+self._ca(self.norm2_s(shape),cond,self.ca_q_s,self.ca_kv_s,self.ca_o_s)
        pose=pose+self._ca(self.norm2_p(pose),cond,self.ca_q_p,self.ca_kv_p,self.ca_o_p)
        # MLP
        hs_=self._ln(shape)*(1+sc_f.unsqueeze(1))+sh_f.unsqueeze(1)
        hp_=self._ln(pose)*(1+sc_f.unsqueeze(1))+sh_f.unsqueeze(1)
        shape=shape+self.mlp_s(hs_)*g_f.unsqueeze(1)
        pose=pose+self.mlp_p(hp_)*g_f.unsqueeze(1)
        return shape, pose

    def load_sd(self, sd: dict, pfx: str):
        p=pfx
        self.adaLN[1].weight.data.copy_(sd[p+"adaLN_modulation.1.weight"])
        self.adaLN[1].bias.data.copy_(sd[p+"adaLN_modulation.1.bias"])
        self.norm2_s.weight.data.copy_(sd[p+"norm2.shape.weight"])
        self.norm2_s.bias.data.copy_(sd[p+"norm2.shape.bias"])
        self.norm2_p.weight.data.copy_(sd[p+"norm2.6drotation_normalized.weight"])
        self.norm2_p.bias.data.copy_(sd[p+"norm2.6drotation_normalized.bias"])
        for attr,key in [(self.sa_qkv_s,"self_attn.to_qkv.shape"),(self.sa_qkv_p,"self_attn.to_qkv.6drotation_normalized"),
                         (self.sa_out_s,"self_attn.to_out.shape"),(self.sa_out_p,"self_attn.to_out.6drotation_normalized"),
                         (self.ca_q_s,"cross_attn.shape.to_q"),(self.ca_kv_s,"cross_attn.shape.to_kv"),(self.ca_o_s,"cross_attn.shape.to_out"),
                         (self.ca_q_p,"cross_attn.6drotation_normalized.to_q"),(self.ca_kv_p,"cross_attn.6drotation_normalized.to_kv"),(self.ca_o_p,"cross_attn.6drotation_normalized.to_out")]:
            attr.weight.data.copy_(sd[p+key+".weight"])
            attr.bias.data.copy_(sd[p+key+".bias"])
        for rms,key in [(self.sa_qrms_s,"self_attn.q_rms_norm.shape"),(self.sa_krms_s,"self_attn.k_rms_norm.shape"),
                        (self.sa_qrms_p,"self_attn.q_rms_norm.6drotation_normalized"),(self.sa_krms_p,"self_attn.k_rms_norm.6drotation_normalized")]:
            rms.gamma.data.copy_(sd[p+key+".gamma"])
        for mlp,lat in [(self.mlp_s,"shape"),(self.mlp_p,"6drotation_normalized")]:
            mlp[0].weight.data.copy_(sd[p+f"mlp.{lat}.mlp.0.weight"])
            mlp[0].bias.data.copy_(sd[p+f"mlp.{lat}.mlp.0.bias"])
            mlp[2].weight.data.copy_(sd[p+f"mlp.{lat}.mlp.2.weight"])
            mlp[2].bias.data.copy_(sd[p+f"mlp.{lat}.mlp.2.bias"])

class SSBackbone(nn.Module):
    def __init__(self, blocks, t_emb, d_emb):
        super().__init__()
        self.blocks=nn.ModuleList(blocks); self.t_emb=t_emb; self.d_emb=d_emb
    def forward(self,shape,pose,t,d,cond):
        te=self.t_emb(t)+self.d_emb(d)
        for blk in self.blocks:
            shape,pose=blk(shape,pose,te,cond)
        return shape, pose

# ===========================================================================
# SS Latent projections
# ===========================================================================
class SSLatentInAll(nn.Module):
    """Merged input projections for all 5 SS latent modalities."""
    def __init__(self):
        super().__init__()
        MC = SS_MODEL_CHANNELS
        self.lin_shape = nn.Linear(8, MC)
        self.pos_shape = nn.Parameter(torch.zeros(4096, MC))
        self.lin_rot = nn.Linear(6, MC)
        self.pos_rot = nn.Parameter(torch.zeros(1, MC))
        self.lin_trans = nn.Linear(3, MC)
        self.pos_trans = nn.Parameter(torch.zeros(1, MC))
        self.lin_scale = nn.Linear(3, MC)
        self.pos_scale = nn.Parameter(torch.zeros(1, MC))
        self.lin_tscale = nn.Linear(1, MC)
        self.pos_tscale = nn.Parameter(torch.zeros(1, MC))

    def forward(self, x_shape, x_rot, x_trans, x_scale, x_tscale):
        return (
            self.lin_shape(x_shape) + self.pos_shape.unsqueeze(0),
            self.lin_rot(x_rot) + self.pos_rot.unsqueeze(0),
            self.lin_trans(x_trans) + self.pos_trans.unsqueeze(0),
            self.lin_scale(x_scale) + self.pos_scale.unsqueeze(0),
            self.lin_tscale(x_tscale) + self.pos_tscale.unsqueeze(0),
        )

class SSLatentOutAll(nn.Module):
    """Merged output projections for all 5 SS latent modalities."""
    def __init__(self):
        super().__init__()
        MC = SS_MODEL_CHANNELS
        self.m_ch = MC
        self.lin_shape = nn.Linear(MC, 8)
        self.lin_rot = nn.Linear(MC, 6)
        self.lin_trans = nn.Linear(MC, 3)
        self.lin_scale = nn.Linear(MC, 3)
        self.lin_tscale = nn.Linear(MC, 1)

    def forward(self, h_shape, h_rot, h_trans, h_scale, h_tscale):
        mc = self.m_ch
        return (
            self.lin_shape(F.layer_norm(h_shape, [mc])),
            self.lin_rot(F.layer_norm(h_rot, [mc])),
            self.lin_trans(F.layer_norm(h_trans, [mc])),
            self.lin_scale(F.layer_norm(h_scale, [mc])),
            self.lin_tscale(F.layer_norm(h_tscale, [mc])),
        )

class SLATTimestepEmbedAll(nn.Module):
    """Merged: slat_t_embedder + 4 IO block emb_layers.
    Input: scalar t
    Outputs: t_emb (1024), emb_in0 (256), emb_in1 (2048), emb_out0 (2048), emb_out1 (256)
    """
    def __init__(self, t_emb_module, emb_in0, emb_in1, emb_out0, emb_out1):
        super().__init__()
        self.t_emb = t_emb_module
        self.emb_in0 = emb_in0    # SiLU + Linear(1024 → 256)
        self.emb_in1 = emb_in1    # SiLU + Linear(1024 → 2048)
        self.emb_out0 = emb_out0  # SiLU + Linear(1024 → 2048)
        self.emb_out1 = emb_out1  # SiLU + Linear(1024 → 256)

    def forward(self, t):
        t_emb = self.t_emb(t)
        te = F.silu(t_emb)
        return (t_emb,
                self.emb_in0(te),
                self.emb_in1(te),
                self.emb_out0(te),
                self.emb_out1(te))

# ===========================================================================
# SS Decoder (Conv3D VAE)
# ===========================================================================
class ResBlock3d(nn.Module):
    def __init__(self,ch,out=None):
        super().__init__()
        out=out or ch
        self.norm1=ChannelLayerNorm32(ch); self.norm2=ChannelLayerNorm32(out)
        self.conv1=nn.Conv3d(ch,out,3,padding=1); self.conv2=nn.Conv3d(out,out,3,padding=1)
        self.skip=nn.Conv3d(ch,out,1) if ch!=out else nn.Identity()
    def forward(self,x):
        h=self.conv1(F.silu(self.norm1(x))); h=self.conv2(F.silu(self.norm2(h)))
        return h+self.skip(x)

class UpBlock3d(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv=nn.Conv3d(in_ch,out_ch*8,3,padding=1)
    def forward(self,x):
        x=self.conv(x); B,C8,D,H,W=x.shape; C=C8//8
        x=x.view(B,C,2,2,2,D,H,W).permute(0,1,5,2,6,3,7,4).contiguous()
        return x.view(B,C,D*2,H*2,W*2)

class SSDecoder(nn.Module):
    """Matches ss_decoder.ckpt: input_layer, middle_block, blocks, out_layer."""
    def __init__(self, out_ch=1, lat_ch=8, chs=(512,128,32), nres=2, nmid=2):
        super().__init__()
        self.input_layer = nn.Conv3d(lat_ch, chs[0], 3, padding=1)
        self.middle_block = nn.Sequential(*[ResBlock3d(chs[0]) for _ in range(nmid)])
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(chs):
            for _ in range(nres): self.blocks.append(ResBlock3d(ch))
            if i < len(chs)-1: self.blocks.append(UpBlock3d(ch, chs[i+1]))
        self.out_layer = nn.Sequential(
            ChannelLayerNorm32(chs[-1]),
            nn.SiLU(),
            nn.Conv3d(chs[-1], out_ch, 3, padding=1),
        )
    def forward(self, x):
        h = self.middle_block(self.input_layer(x))
        for b in self.blocks: h = b(h)
        return self.out_layer(h)

# ===========================================================================
# SLAT Block (dense attention on sparse voxel features)
# ===========================================================================
class SLATBlock(nn.Module):
    def __init__(self, ch: int, ctx_ch: int, heads: int):
        super().__init__()
        self.ch=ch; self.H=heads; self.hd=ch//heads
        self.norm2=LayerNorm32(ch,elementwise_affine=True,eps=1e-6)
        self.adaLN=nn.Sequential(nn.SiLU(),nn.Linear(ch,6*ch,bias=True))
        self.sa_qkv=nn.Linear(ch,ch*3,bias=True)
        self.sa_qrms=MultiHeadRMSNorm(self.hd,heads)
        self.sa_krms=MultiHeadRMSNorm(self.hd,heads)
        self.sa_out=nn.Linear(ch,ch,bias=True)
        self.ca_q=nn.Linear(ch,ch,bias=True)
        self.ca_kv=nn.Linear(ctx_ch,ch*2,bias=True)
        self.ca_out=nn.Linear(ch,ch,bias=True)
        mlp_h=ch*4
        self.mlp=nn.Sequential(nn.Linear(ch,mlp_h),nn.GELU(approximate="tanh"),nn.Linear(mlp_h,ch))

    def _ln(self,x): return F.layer_norm(x.float(),[self.ch]).to(x.dtype)

    def forward(self, feats: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor):
        N,C = feats.shape
        sh_m,sc_m,g_m,sh_f,sc_f,g_f = self.adaLN(t_emb).chunk(6,dim=1)  # each (1,C)
        # SA
        h=self._ln(feats)*(1+sc_m)+sh_m
        qkv=self.sa_qkv(h).reshape(N,3,self.H,self.hd)
        q,k,v=qkv.unbind(1)
        q=self.sa_qrms(q); k=self.sa_krms(k)
        q=q.unsqueeze(0).permute(0,2,1,3); k=k.unsqueeze(0).permute(0,2,1,3); v=v.unsqueeze(0).permute(0,2,1,3)
        h_sa=F.scaled_dot_product_attention(q,k,v).permute(0,2,1,3).reshape(1,N,C).squeeze(0)
        feats=feats+self.sa_out(h_sa)*g_m
        # CA
        h_ca=self.norm2(feats).unsqueeze(0)  # (1,N,C)
        q_ca=self.ca_q(h_ca).reshape(1,N,self.H,self.hd).permute(0,2,1,3)
        kv=self.ca_kv(cond).reshape(1,cond.shape[1],2,self.H,self.hd)
        k_ca,v_ca=kv.unbind(2); k_ca=k_ca.permute(0,2,1,3); v_ca=v_ca.permute(0,2,1,3)
        h_ca=F.scaled_dot_product_attention(q_ca,k_ca,v_ca).permute(0,2,1,3).reshape(1,N,C).squeeze(0)
        feats=feats+self.ca_out(h_ca)
        # MLP
        h_m=self._ln(feats)*(1+sc_f)+sh_f
        feats=feats+self.mlp(h_m)*g_f
        return feats

    def load_sd(self,sd,pfx):
        p=pfx
        self.adaLN[1].weight.data.copy_(sd[p+"adaLN_modulation.1.weight"])
        self.adaLN[1].bias.data.copy_(sd[p+"adaLN_modulation.1.bias"])
        self.norm2.weight.data.copy_(sd[p+"norm2.weight"])
        self.norm2.bias.data.copy_(sd[p+"norm2.bias"])
        for attr,key in [(self.sa_qkv,"self_attn.to_qkv"),(self.sa_out,"self_attn.to_out"),
                         (self.ca_q,"cross_attn.to_q"),(self.ca_kv,"cross_attn.to_kv"),(self.ca_out,"cross_attn.to_out")]:
            attr.weight.data.copy_(sd[p+key+".weight"])
            attr.bias.data.copy_(sd[p+key+".bias"])
        self.sa_qrms.gamma.data.copy_(sd[p+"self_attn.q_rms_norm.gamma"])
        self.sa_krms.gamma.data.copy_(sd[p+"self_attn.k_rms_norm.gamma"])
        self.mlp[0].weight.data.copy_(sd[p+"mlp.mlp.0.weight"])
        self.mlp[0].bias.data.copy_(sd[p+"mlp.mlp.0.bias"])
        self.mlp[2].weight.data.copy_(sd[p+"mlp.mlp.2.weight"])
        self.mlp[2].bias.data.copy_(sd[p+"mlp.mlp.2.bias"])

class SLATAttnBlocks(nn.Module):
    def __init__(self, blocks, t_emb):
        super().__init__()
        self.blocks=nn.ModuleList(blocks); self.t_emb=t_emb
        freq_dim = SLAT_MODEL_CHANNELS // 3 // 2
        freqs = 1.0/(10000**(torch.arange(freq_dim, dtype=torch.float32)/freq_dim))
        self.register_buffer("freqs", freqs)

    def _ape(self, coords: torch.Tensor) -> torch.Tensor:
        N=coords.shape[0]; C=SLAT_MODEL_CHANNELS
        flat=coords.reshape(-1).float()
        phases=torch.outer(flat, self.freqs.float())
        emb=torch.cat([torch.sin(phases),torch.cos(phases)],dim=-1).reshape(N,-1)
        if emb.shape[1]<C:
            emb=torch.cat([emb,torch.zeros(N,C-emb.shape[1],device=emb.device)],dim=-1)
        return emb[:,:C]

    def forward(self, feats: torch.Tensor, coords: torch.Tensor,
                t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        te=self.t_emb(t)
        feats=feats+self._ape(coords).to(feats.dtype)
        for blk in self.blocks:
            feats=blk(feats, te, cond)
        return feats

# ---------------------------------------------------------------------------
# Decoder block: self-attention + MLP only (no adaln, no cross-attention)
# ---------------------------------------------------------------------------
class SLATDecoderBlock(nn.Module):
    """Sparse transformer block for latent decoder: SA + MLP, no conditioning."""
    def __init__(self, ch: int, heads: int):
        super().__init__()
        self.ch = ch; self.H = heads; self.hd = ch // heads
        self.sa_qkv = nn.Linear(ch, ch * 3, bias=True)
        self.sa_out = nn.Linear(ch, ch, bias=True)
        mlp_h = ch * 4
        self.mlp = nn.Sequential(
            nn.Linear(ch, mlp_h),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_h, ch),
        )

    def _ln(self, x):
        return F.layer_norm(x.float(), [self.ch]).to(x.dtype)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        N, C = feats.shape
        # Self-attention with pre-norm (non-learnable layer norm)
        h = self._ln(feats)
        qkv = self.sa_qkv(h).reshape(N, 3, self.H, self.hd)
        q, k, v = qkv.unbind(1)
        q = q.unsqueeze(0).permute(0, 2, 1, 3)
        k = k.unsqueeze(0).permute(0, 2, 1, 3)
        v = v.unsqueeze(0).permute(0, 2, 1, 3)
        h = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(N, C)
        feats = feats + self.sa_out(h)
        # MLP with pre-norm
        h = self._ln(feats)
        feats = feats + self.mlp(h)
        return feats

    def load_sd(self, sd, pfx):
        for attr, key in [
            (self.sa_qkv, "attn.to_qkv"),
            (self.sa_out, "attn.to_out"),
        ]:
            attr.weight.data.copy_(sd[pfx + key + ".weight"])
            attr.bias.data.copy_(sd[pfx + key + ".bias"])
        self.mlp[0].weight.data.copy_(sd[pfx + "mlp.mlp.0.weight"])
        self.mlp[0].bias.data.copy_(sd[pfx + "mlp.mlp.0.bias"])
        self.mlp[2].weight.data.copy_(sd[pfx + "mlp.mlp.2.weight"])
        self.mlp[2].bias.data.copy_(sd[pfx + "mlp.mlp.2.bias"])


class SLATDecoderModel(nn.Module):
    """Full latent decoder: input_layer + APE + N blocks + layer_norm + out_layer.

    The transformer blocks operate on dense (N_vox, C) tensors — swin windowing
    is approximated as full attention (same approach as slat_attn_blocks export).
    """
    def __init__(self, blocks: list, in_channels: int, model_channels: int,
                 out_channels: int):
        super().__init__()
        self.model_channels = model_channels
        self.in_lin = nn.Linear(in_channels, model_channels, bias=True)
        self.blocks = nn.ModuleList(blocks)
        self.out_lin = nn.Linear(model_channels, out_channels, bias=True)
        freq_dim = model_channels // 3 // 2
        freqs = 1.0 / (10000 ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim))
        self.register_buffer("freqs", freqs)

    def _ape(self, coords: torch.Tensor) -> torch.Tensor:
        N = coords.shape[0]; C = self.model_channels
        flat = coords.reshape(-1).float()
        phases = torch.outer(flat, self.freqs)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1).reshape(N, -1)
        if emb.shape[1] < C:
            emb = torch.cat([emb, torch.zeros(N, C - emb.shape[1], device=emb.device)], dim=-1)
        return emb[:, :C]

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        feats = self.in_lin(feats)
        feats = feats + self._ape(coords).to(feats.dtype)
        for blk in self.blocks:
            feats = blk(feats)
        # Final layer norm (non-learnable, matching baseline decoder)
        feats = F.layer_norm(feats.float(), feats.shape[-1:]).to(feats.dtype)
        feats = self.out_lin(feats)
        return feats

    def load_sd(self, sd: dict):
        self.in_lin.weight.data.copy_(sd["input_layer.weight"])
        self.in_lin.bias.data.copy_(sd["input_layer.bias"])
        for i, blk in enumerate(self.blocks):
            blk.load_sd(sd, f"blocks.{i}.")
        self.out_lin.weight.data.copy_(sd["out_layer.weight"])
        self.out_lin.bias.data.copy_(sd["out_layer.bias"])


class SLATDecoderWindowedModel(nn.Module):
    """Single-graph Swin-windowed GS decoder (replaces 14 per-block/io XMLs).

    Windowing (window_size=8, alternating shift 0/4 on even/odd blocks) is done
    inside the OV graph via precomputed gather / inverse-gather index tensors
    supplied at inference time. Each block gathers voxels into padded windows
    (num_win, max_occ, C), runs masked self-attention (padding keys masked out
    → numerically identical to per-window full attention), then scatters back
    via the inverse permutation. All 12 blocks + input/APE/output live in one
    graph, so the entire decoder is a single OV model call.

    Inputs (dynamic N, and per-parity window shapes):
        feats:   (N, 8)   float32 — raw SLAT latent features
        coords:  (N, 3)   float32 — voxel zyx coords (for APE)
        perm_e:  (Le,)    int64   — even-block gather indices into [0..N] (N=pad row)
        inv_e:   (N,)     int64   — even-block inverse (voxel → slot in flattened windows)
        mask_e:  (nwe, moe) float32 — even-block additive attn mask (0 valid, -3e4 pad)
        perm_o/inv_o/mask_o: odd-block (shift=4) counterparts
    """
    def __init__(self, blocks, in_channels, model_channels, out_channels):
        super().__init__()
        self.C = model_channels
        self.in_lin = nn.Linear(in_channels, model_channels, bias=True)
        self.blocks = nn.ModuleList(blocks)
        self.out_lin = nn.Linear(model_channels, out_channels, bias=True)
        freq_dim = model_channels // 3 // 2
        freqs = 1.0 / (10000 ** (torch.arange(freq_dim, dtype=torch.float32) / freq_dim))
        self.register_buffer("freqs", freqs)

    def _ape(self, coords):
        N = coords.shape[0]; C = self.C
        flat = coords.reshape(-1).float()
        phases = torch.outer(flat, self.freqs)
        emb = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1).reshape(N, -1)
        if emb.shape[1] < C:
            emb = torch.cat([emb, torch.zeros(N, C - emb.shape[1], device=emb.device)], dim=-1)
        return emb[:, :C]

    def _block_windowed(self, feats, blk, perm, inv, addmask):
        # feats: (N, C). Append a zero pad row so perm can reference index N.
        C = self.C
        feats_pad = torch.cat([feats, feats.new_zeros(1, C)], dim=0)   # (N+1, C)
        g = feats_pad.index_select(0, perm)                            # (L, C)
        nw = addmask.shape[0]; mo = addmask.shape[1]
        g = g.reshape(nw, mo, C)                                       # (nw, mo, C)
        # ── self-attention (pre-norm), windowed with additive key mask ──
        h = F.layer_norm(g.float(), [C]).to(g.dtype)
        H = blk.H; hd = blk.hd
        qkv = blk.sa_qkv(h).reshape(nw, mo, 3, H, hd)
        q, k, v = qkv.unbind(2)                                        # (nw, mo, H, hd)
        q = q.permute(0, 2, 1, 3); k = k.permute(0, 2, 1, 3); v = v.permute(0, 2, 1, 3)
        am = addmask.reshape(nw, 1, 1, mo)                             # broadcast over H,queries
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)      # (nw, H, mo, hd)
        a = a.permute(0, 2, 1, 3).reshape(nw, mo, C)
        g = g + blk.sa_out(a)
        # ── MLP (pre-norm) ──
        h = F.layer_norm(g.float(), [C]).to(g.dtype)
        g = g + blk.mlp(h)
        # ── scatter back: flatten windows, gather valid slots per voxel ──
        flat = g.reshape(nw * mo, C)                                   # (L, C)
        return flat.index_select(0, inv)                              # (N, C)

    def forward(self, feats, coords, perm_e, inv_e, mask_e, perm_o, inv_o, mask_o):
        feats = self.in_lin(feats)
        feats = feats + self._ape(coords).to(feats.dtype)
        for i, blk in enumerate(self.blocks):
            if i % 2 == 0:
                feats = self._block_windowed(feats, blk, perm_e, inv_e, mask_e)
            else:
                feats = self._block_windowed(feats, blk, perm_o, inv_o, mask_o)
        feats = F.layer_norm(feats.float(), feats.shape[-1:]).to(feats.dtype)
        return self.out_lin(feats)

    def load_sd(self, sd):
        self.in_lin.weight.data.copy_(sd["input_layer.weight"])
        self.in_lin.bias.data.copy_(sd["input_layer.bias"])
        for i, blk in enumerate(self.blocks):
            blk.load_sd(sd, f"blocks.{i}.")
        self.out_lin.weight.data.copy_(sd["out_layer.weight"])
        self.out_lin.bias.data.copy_(sd["out_layer.bias"])


# ---------------------------------------------------------------------------
# PointPatchEmbed outer model (3-D point projection, no inner ViT here)
# ---------------------------------------------------------------------------
class PointPatchOuterModel(nn.Module):
    """Projects per-pixel xyz to embed_dim; replaces invalid pixels with learned token.

    Input:
        xyz:        (N_px, 3)  float32  — per-pixel xyz after resize (invalid=0)
        valid_mask: (N_px, 1)  float32  — 1=valid, 0=invalid

    Output:
        proj_feats: (N_px, embed_dim) float32
    """
    def __init__(self, point_proj_weight: torch.Tensor,
                 point_proj_bias: torch.Tensor,
                 invalid_xyz_token: torch.Tensor):
        super().__init__()
        embed_dim = point_proj_weight.shape[0]
        self.proj = nn.Linear(3, embed_dim, bias=True)
        self.proj.weight.data.copy_(point_proj_weight)
        self.proj.bias.data.copy_(point_proj_bias)
        self.register_buffer("invalid_token", invalid_xyz_token)   # (embed_dim,)

    def forward(self, xyz: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        projected = self.proj(xyz)                                   # (N_px, D)
        inv = self.invalid_token.unsqueeze(0).expand_as(projected)  # (N_px, D)
        return projected * valid_mask + inv * (1.0 - valid_mask)


# ---------------------------------------------------------------------------
# PointPatchEmbed FULL model (outer proj + inner ViT window attention)
# Replicates sam3d_objects PointPatchEmbed.forward exactly (remap='linear').
# Standard timm ViT Block re-implemented inline (deterministic at inference).
# ---------------------------------------------------------------------------
class _PPViTAttention(nn.Module):
    """Matches timm.models.vision_transformer.Attention (qk_norm=False, no drop)."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _PPViTMlp(nn.Module):
    """Matches timm Mlp (act=GELU exact)."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _PPViTBlock(nn.Module):
    """Matches timm Block (ls=Identity, drop_path=Identity, norm eps=1e-6)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _PPViTAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _PPViTMlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PointPatchFullModel(nn.Module):
    """Full PointPatchEmbed: xyz (nan-free, invalid=0) + valid mask -> window tokens.

    Input:
        xyz:   (1, 3, H, W)  float32  — SSI-normalized pointmap, invalid pixels set to 0
        valid: (1, 1, H, W)  float32  — 1=valid, 0=invalid
    Output:
        tokens: (1, 1024, 512)
    """
    def __init__(self, sd: dict, pfx: str,
                 input_size: int = 256, patch_size: int = 8,
                 embed_dim: int = 512, num_heads: int = 16, mlp_ratio: float = 2.0):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.np_side = input_size // patch_size  # 32

        self.point_proj = nn.Linear(3, embed_dim, bias=True)
        self.point_proj.weight.data.copy_(sd[pfx + "point_proj.weight"].float())
        self.point_proj.bias.data.copy_(sd[pfx + "point_proj.bias"].float())
        self.register_buffer("invalid_xyz_token", sd[pfx + "invalid_xyz_token"].float())
        self.register_buffer("cls_token", sd[pfx + "cls_token"].float())            # (1,1,512)
        self.register_buffer("pos_embed_window", sd[pfx + "pos_embed_window"].float())  # (1,65,512)
        self.register_buffer("pos_embed", sd[pfx + "pos_embed"].float())            # (1,512,32,32)

        self.block = _PPViTBlock(embed_dim, num_heads, mlp_ratio)
        self.block.norm1.weight.data.copy_(sd[pfx + "blocks.0.norm1.weight"].float())
        self.block.norm1.bias.data.copy_(sd[pfx + "blocks.0.norm1.bias"].float())
        self.block.attn.qkv.weight.data.copy_(sd[pfx + "blocks.0.attn.qkv.weight"].float())
        self.block.attn.qkv.bias.data.copy_(sd[pfx + "blocks.0.attn.qkv.bias"].float())
        self.block.attn.proj.weight.data.copy_(sd[pfx + "blocks.0.attn.proj.weight"].float())
        self.block.attn.proj.bias.data.copy_(sd[pfx + "blocks.0.attn.proj.bias"].float())
        self.block.norm2.weight.data.copy_(sd[pfx + "blocks.0.norm2.weight"].float())
        self.block.norm2.bias.data.copy_(sd[pfx + "blocks.0.norm2.bias"].float())
        self.block.mlp.fc1.weight.data.copy_(sd[pfx + "blocks.0.mlp.fc1.weight"].float())
        self.block.mlp.fc1.bias.data.copy_(sd[pfx + "blocks.0.mlp.fc1.bias"].float())
        self.block.mlp.fc2.weight.data.copy_(sd[pfx + "blocks.0.mlp.fc2.weight"].float())
        self.block.mlp.fc2.bias.data.copy_(sd[pfx + "blocks.0.mlp.fc2.bias"].float())

    def forward(self, xyz: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        S = self.input_size
        xyz = F.interpolate(xyz, size=S, mode="nearest")       # (1,3,S,S)
        valid = F.interpolate(valid, size=S, mode="nearest")   # (1,1,S,S)
        xyz = xyz.permute(0, 2, 3, 1)                           # (1,S,S,3)
        valid = valid.permute(0, 2, 3, 1)                       # (1,S,S,1)
        # remap='linear' == identity; project + invalid token
        x = self.point_proj(xyz)                               # (1,S,S,512)
        x = x * valid + self.invalid_xyz_token.view(1, 1, 1, -1) * (1.0 - valid)

        B = 1
        p = self.patch_size
        nP = self.np_side
        x = x.view(B, nP, p, nP, p, self.embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, p * p, self.embed_dim)  # (B*1024,64,512)
        cls = self.cls_token.expand(x.shape[0], -1, -1)                     # (nW,1,512)
        toks = torch.cat([cls, x], dim=1) + self.pos_embed_window           # (nW,65,512)
        toks = self.block(toks)
        we = toks[:, 0].view(B, nP * nP, self.embed_dim)                    # (1,1024,512)
        pos = self.pos_embed.permute(0, 2, 3, 1).reshape(1, nP * nP, self.embed_dim)
        return we + pos


# ===========================================================================
# Export functions
# ===========================================================================

COND_PFX_SS = "_base_models.condition_embedder."
COND_PFX_SLAT = "_base_models.condition_embedder."
BB_PFX = "_base_models.generator.reverse_fn.backbone."

def _build_t_emb(sd, prefix, channels):
    t=TimestepEmbedder(channels)
    t.mlp[0].weight.data.copy_(sd[prefix+"mlp.0.weight"])
    t.mlp[0].bias.data.copy_(sd[prefix+"mlp.0.bias"])
    t.mlp[2].weight.data.copy_(sd[prefix+"mlp.2.weight"])
    t.mlp[2].bias.data.copy_(sd[prefix+"mlp.2.bias"])
    return t

def export_dino(full_sd, cond_pfx, idx, prenorm, outdir, name):
    bb=load_dino_backbone(full_sd, cond_pfx, idx)
    w=DinoWrapper(bb, prenorm=prenorm)
    inp=torch.zeros(1,3,DINO_INPUT_SIZE,DINO_INPUT_SIZE)
    return torch_to_ov(w,inp,outdir,name)

def export_ss_dino_image(sd, outdir):
    print("  [SS] DINOv2 image backbone...")
    return export_dino(sd,COND_PFX_SS,0,False,outdir,"ss_dino_image")

def export_ss_dino_mask(sd, outdir):
    print("  [SS] DINOv2 mask backbone (1ch input)...")
    bb=load_dino_backbone(sd,COND_PFX_SS,1)
    w=DinoWrapper(bb,prenorm=False)
    inp=torch.zeros(1,1,DINO_INPUT_SIZE,DINO_INPUT_SIZE)
    return torch_to_ov(w,inp,outdir,"ss_dino_mask")

def export_proj_net(sd, pfx, in_d, out_d, hidden_d, outdir, name):
    net=ProjectionNet(in_d,out_d,hidden_d)
    net.load_sd(sd, pfx)
    inp=torch.zeros(1,1374,in_d)
    torch_to_ov(net,inp,outdir,name)

def export_ss_embedder_projections(sd, outdir):
    print("  [SS] EmbedderFuser projection nets...")
    for i in range(2):
        export_proj_net(sd, COND_PFX_SS+f"projection_nets.{i}.",
                        COND_EMBED_DIM, COND_EMBED_DIM, 2816, outdir, f"ss_embedder_proj_{i}")

def export_ss_backbone(sd, outdir):
    print("  [SS] SS backbone (24 MOT blocks)...")
    bb_sd=filter_prefix(sd, BB_PFX)
    blocks=[SSMOTBlock(SS_MODEL_CHANNELS,COND_EMBED_DIM,SS_NUM_HEADS) for _ in range(SS_NUM_BLOCKS)]
    for i,b in enumerate(blocks):
        b.load_sd(bb_sd, f"blocks.{i}.")
        b.eval()
    t_e=_build_t_emb(bb_sd,"t_embedder.",SS_MODEL_CHANNELS)
    d_e=_build_t_emb(bb_sd,"d_embedder.",SS_MODEL_CHANNELS)
    model=SSBackbone(blocks,t_e,d_e)
    # Use small example sizes for tracing (model is dynamic)
    B=1; Ls=64; Lp=4; Lc=128
    s=torch.zeros(B,Ls,SS_MODEL_CHANNELS)
    p=torch.zeros(B,Lp,SS_MODEL_CHANNELS)
    t=torch.zeros(B); d=torch.zeros(B)
    c=torch.zeros(B,Lc,COND_EMBED_DIM)
    return torch_to_ov(model,(s,p,t,d,c),outdir,"ss_backbone")

def export_ss_latent_projections(sd, outdir):
    """Export merged SS latent projections: all 5 in → 1 model, all 5 out → 1 model."""
    print("  [SS] SS latent projections (merged all-in-one)...")
    bb_sd=filter_prefix(sd, BB_PFX)

    # ── Merged latent_in_all ──
    m_in = SSLatentInAll()
    for attr, name in [("shape","shape"),("rot","6drotation_normalized"),
                       ("trans","translation"),("scale","scale"),("tscale","translation_scale")]:
        lin = getattr(m_in, f"lin_{attr}")
        pos = f"pos_{attr}"
        lin.weight.data.copy_(bb_sd[f"latent_mapping.{name}.input_layer.weight"])
        lin.bias.data.copy_(bb_sd[f"latent_mapping.{name}.input_layer.bias"])
        getattr(m_in, pos).data.copy_(bb_sd[f"latent_mapping.{name}.pos_emb"])
    m_in.eval()
    B=1
    example_in = (
        torch.zeros(B, 4096, 8),  # shape
        torch.zeros(B, 1, 6),     # rot
        torch.zeros(B, 1, 3),     # trans
        torch.zeros(B, 1, 3),     # scale
        torch.zeros(B, 1, 1),     # tscale
    )
    torch_to_ov(m_in, example_in, outdir, "ss_latent_in_all")

    # ── Merged latent_out_all ──
    m_out = SSLatentOutAll()
    for attr, name in [("shape","shape"),("rot","6drotation_normalized"),
                       ("trans","translation"),("scale","scale"),("tscale","translation_scale")]:
        lin = getattr(m_out, f"lin_{attr}")
        lin.weight.data.copy_(bb_sd[f"latent_mapping.{name}.out_layer.weight"])
        lin.bias.data.copy_(bb_sd[f"latent_mapping.{name}.out_layer.bias"])
    m_out.eval()
    MC = SS_MODEL_CHANNELS
    example_out = (
        torch.zeros(B, 4096, MC),  # h_shape
        torch.zeros(B, 1, MC),     # h_rot
        torch.zeros(B, 1, MC),     # h_trans
        torch.zeros(B, 1, MC),     # h_scale
        torch.zeros(B, 1, MC),     # h_tscale
    )
    torch_to_ov(m_out, example_out, outdir, "ss_latent_out_all")

def export_ss_decoder(ss_dec_ckpt: Path, outdir):
    print("  [SS] SS decoder (Conv3d VAE)...")
    dec_sd=load_checkpoint(ss_dec_ckpt)
    m=SSDecoder()
    missing,_=m.load_state_dict(dec_sd,strict=False)
    if missing: print(f"    Missing keys: {missing[:5]}")
    inp=torch.zeros(1,8,16,16,16)
    torch_to_ov(m,inp,outdir,"ss_decoder")

def export_slat_dinos(slat_sd, outdir):
    print("  [SLAT] DINOv2 backbones (prenorm=True)...")
    for i in range(2):
        export_dino(slat_sd,COND_PFX_SLAT,i,True,outdir,f"slat_dino_{i}")

def export_slat_embedder_projections(slat_sd, outdir):
    print("  [SLAT] EmbedderFuser projection nets...")
    for i in range(2):
        export_proj_net(slat_sd, COND_PFX_SLAT+f"projection_nets.{i}.",
                        COND_EMBED_DIM, COND_EMBED_DIM, 2816, outdir, f"slat_embedder_proj_{i}")

def export_slat_attn_blocks(slat_sd, outdir):
    print("  [SLAT] SLAT attention blocks (24 blocks, dynamic N_vox)...")
    bb_sd=filter_prefix(slat_sd, BB_PFX)
    blocks=[SLATBlock(SLAT_MODEL_CHANNELS,SLAT_COND_CHANNELS,SLAT_NUM_HEADS) for _ in range(SLAT_NUM_BLOCKS)]
    for i,b in enumerate(blocks):
        b.load_sd(bb_sd, f"blocks.{i}.")
        b.eval()
    t_e=_build_t_emb(bb_sd,"t_embedder.",SLAT_MODEL_CHANNELS)
    model=SLATAttnBlocks(blocks,t_e)
    model.eval()
    N=64
    feats=torch.zeros(N,SLAT_MODEL_CHANNELS)
    coords=torch.zeros(N,3,dtype=torch.long)
    t=torch.zeros(1)
    cond=torch.zeros(1,128,SLAT_COND_CHANNELS)
    return torch_to_ov(model,(feats,coords,t,cond),outdir,"slat_attn_blocks")

def export_slat_sparse_io_weights(slat_sd: dict, outdir: Path):
    print("  [SLAT] Saving sparse IO weights for custom extension...")
    bb_sd=filter_prefix(slat_sd, BB_PFX)
    wdir=outdir/"slat_sparse_io_weights"; wdir.mkdir(parents=True, exist_ok=True)
    manifest={"layers":{}}
    def sv(key, t):
        fname=key.replace(".","_")+".npy"
        np.save(str(wdir/fname), t.float().numpy())
        return fname
    # input_layer (SparseLinear = nn.Linear)
    for suf in ["weight","bias"]:
        k=f"input_layer.{suf}"
        manifest["layers"][k]={"file":sv(k,bb_sd[k]),"shape":list(bb_sd[k].shape)}
    # input_blocks (SparseResBlock3d with SparseConv3d)
    for blk in range(2):
        for part in ["conv1","conv2","norm1","emb_layers.1","skip_connection"]:
            for suf in ["weight","bias"]:
                ck=f"input_blocks.{blk}.{part}.conv.{suf}"
                k=f"input_blocks.{blk}.{part}.{suf}"
                if ck in bb_sd:
                    manifest["layers"][ck]={"file":sv(ck,bb_sd[ck]),"shape":list(bb_sd[ck].shape),"type":"sparse_conv3d"}
                elif k in bb_sd:
                    manifest["layers"][k]={"file":sv(k,bb_sd[k]),"shape":list(bb_sd[k].shape)}
    # out_blocks
    for blk in range(2):
        for part in ["conv1","conv2","norm1","emb_layers.1","skip_connection"]:
            for suf in ["weight","bias"]:
                ck=f"out_blocks.{blk}.{part}.conv.{suf}"
                k=f"out_blocks.{blk}.{part}.{suf}"
                if ck in bb_sd:
                    manifest["layers"][ck]={"file":sv(ck,bb_sd[ck]),"shape":list(bb_sd[ck].shape),"type":"sparse_conv3d"}
                elif k in bb_sd:
                    manifest["layers"][k]={"file":sv(k,bb_sd[k]),"shape":list(bb_sd[k].shape)}
    # out_layer
    for suf in ["weight","bias"]:
        k=f"out_layer.{suf}"
        manifest["layers"][k]={"file":sv(k,bb_sd[k]),"shape":list(bb_sd[k].shape)}
    mpath=outdir/"slat_sparse_io_manifest.json"
    with open(str(mpath),"w") as f: json.dump(manifest,f,indent=2)
    print(f"    Saved {len(manifest['layers'])} weight files and manifest")


def export_slat_io_fused_resblock(slat_sd: dict, outdir: Path):
    """Generate the 4 SparseResBlock3d fused-resblock OV IR models.

    Each SLAT input/out block is a SparseResBlock3d:
        h = SiLU(norm1(x))            (affine LayerNorm)
        h = conv1(h)                  (SubMConv3d)
        h = SiLU(norm2(h)*(1+scale)+shift)  (non-affine LN + emb modulation)
        h = conv2(h)                  (SubMConv3d)
        out = skip_connection(x) + h  (identity or SparseLinear)

    The whole block is fused into a single SparseConv3dEngine custom op that
    runs on the GPU via the OpenCL extension (5 layers: LAYERNORM_SILU,
    SUBM_CONV, NORM2_ACT, SUBM_CONV, RESIDUAL_ADD).  Weights are packed into a
    single FP32 buffer ([all_weights | all_scales | all_biases]) and emitted as
    an IR .xml + .bin pair.  This makes the export self-contained — no
    pre-existing exported models are required.
    """
    print("  [SLAT] Generating fused resblock OV models (custom SparseConv3dEngine)...")
    bb_sd = filter_prefix(slat_sd, BB_PFX)

    # Layer-type codes (must match openvino_extensions/sparse_conv_3d hpp)
    LAYER_SUBM_CONV      = 0
    LAYER_RESIDUAL_ADD   = 7
    LAYER_LAYERNORM_SILU = 6
    LAYER_NORM2_ACT      = 8
    MAX_VOXELS = 65536
    SPATIAL_RES = 64
    NK = 27  # 3x3x3 kernel

    def _t(k):
        return bb_sd[k].float().numpy().astype(np.float32)

    for bn in ["input_blocks.0", "input_blocks.1", "out_blocks.0", "out_blocks.1"]:
        safe = bn.replace(".", "_")
        conv1_w = _t(f"{bn}.conv1.conv.weight")   # (C_mid, 3,3,3, C_in)
        conv1_b = _t(f"{bn}.conv1.conv.bias")     # (C_mid,)
        conv2_w = _t(f"{bn}.conv2.conv.weight")   # (C_out, 3,3,3, C_mid)
        conv2_b = _t(f"{bn}.conv2.conv.bias")     # (C_out,)
        norm1_w = _t(f"{bn}.norm1.weight")        # (C_in,)
        norm1_b = _t(f"{bn}.norm1.bias")          # (C_in,)

        C_mid = conv1_w.shape[0]
        C_in  = conv1_w.shape[-1]
        C_out = conv2_w.shape[0]

        has_skip = f"{bn}.skip_connection.weight" in bb_sd
        if has_skip:
            skip_w = _t(f"{bn}.skip_connection.weight")   # (C_out, C_in)
            skip_b = _t(f"{bn}.skip_connection.bias")     # (C_out,)

        # Layer definitions: [type, cin, cout, nk] x 5
        layer_defs = [
            (LAYER_LAYERNORM_SILU, C_in,  C_in,  0),
            (LAYER_SUBM_CONV,      C_in,  C_mid, NK),
            (LAYER_NORM2_ACT,      C_mid, C_mid, 0),
            (LAYER_SUBM_CONV,      C_mid, C_out, NK),
            (LAYER_RESIDUAL_ADD,   C_in,  C_out, 1 if has_skip else 0),
        ]
        couts = [ld[2] for ld in layer_defs]

        # Pack weights: [all_weights | all_scales | all_biases]
        # weights per layer (only conv/skip layers carry weights)
        weights = [conv1_w.reshape(-1), conv2_w.reshape(-1)]
        if has_skip:
            # SparseLinear weight is (C_out, C_in); engine expects (C_in, C_out)
            weights.append(skip_w.T.reshape(-1))
        # scales per layer (len = layer cout); C++ reads L0 (norm1 gamma) and
        # ones for conv layers; L2/L4 slots are unused placeholders (zeros).
        scales = [
            norm1_w,                            # L0 LAYERNORM_SILU gamma
            np.ones(couts[1], np.float32),      # L1 conv1
            np.zeros(couts[2], np.float32),     # L2 norm2_act (unused)
            np.ones(couts[3], np.float32),      # L3 conv2
            np.zeros(couts[4], np.float32),     # L4 residual (unused)
        ]
        # biases per layer (len = layer cout)
        biases = [
            norm1_b,                            # L0 LAYERNORM_SILU beta
            conv1_b,                            # L1 conv1 bias
            np.zeros(couts[2], np.float32),     # L2 norm2_act (unused)
            conv2_b,                            # L3 conv2 bias
            (skip_b if has_skip                 # L4 skip bias (or zeros)
             else np.zeros(couts[4], np.float32)),
        ]
        packed = np.concatenate(weights + scales + biases).astype(np.float32)
        n_params = int(packed.size)

        # Write weights binary
        bin_path = outdir / f"slat_io_{safe}_fused_resblock.bin"
        packed.tofile(str(bin_path))

        layer_defs_flat = ",".join(str(x) for ld in layer_defs for x in ld)
        xml = f"""<?xml version="1.0"?>
<net name="slat_io_{safe}_fused_resblock" version="11">
  <layers>
    <layer id="0" name="features" type="Parameter" version="opset1">
      <data shape="{MAX_VOXELS},{C_in}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="features">
        <dim>{MAX_VOXELS}</dim><dim>{C_in}</dim>
      </port></output>
    </layer>
    <layer id="1" name="coords" type="Parameter" version="opset1">
      <data shape="{MAX_VOXELS},4" element_type="i32"/>
      <output><port id="0" precision="I32" names="coords">
        <dim>{MAX_VOXELS}</dim><dim>4</dim>
      </port></output>
    </layer>
    <layer id="2" name="num_voxels" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_voxels">
        <dim>1</dim>
      </port></output>
    </layer>
    <layer id="3" name="params" type="Const" version="opset1">
      <data element_type="f32" shape="{n_params}" offset="0" size="{n_params * 4}"/>
      <output><port id="0" precision="FP32">
        <dim>{n_params}</dim>
      </port></output>
    </layer>
    <layer id="4" name="emb_scale" type="Parameter" version="opset1">
      <data shape="{C_mid}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="emb_scale">
        <dim>{C_mid}</dim>
      </port></output>
    </layer>
    <layer id="5" name="emb_shift" type="Parameter" version="opset1">
      <data shape="{C_mid}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="emb_shift">
        <dim>{C_mid}</dim>
      </port></output>
    </layer>
    <layer id="6" name="fused_resblock" type="SparseConv3dEngine" version="sam3d">
      <data max_voxels="{MAX_VOXELS}" in_channels="{C_in}" out_channels="{C_out}"
            num_layers="5" spatial_resolution="{SPATIAL_RES}"
            layer_defs="{layer_defs_flat}"/>
      <input>
        <port id="0"><dim>{MAX_VOXELS}</dim><dim>{C_in}</dim></port>
        <port id="1"><dim>{MAX_VOXELS}</dim><dim>4</dim></port>
        <port id="2"><dim>1</dim></port>
        <port id="3"><dim>{n_params}</dim></port>
        <port id="4"><dim>{C_mid}</dim></port>
        <port id="5"><dim>{C_mid}</dim></port>
      </input>
      <output><port id="6" precision="FP32" names="out_features">
        <dim>{MAX_VOXELS}</dim><dim>{C_out}</dim>
      </port></output>
    </layer>
    <layer id="7" name="output" type="Result" version="opset1">
      <input><port id="0">
        <dim>{MAX_VOXELS}</dim><dim>{C_out}</dim>
      </port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="6" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="6" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="6" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="6" to-port="3"/>
    <edge from-layer="4" from-port="0" to-layer="6" to-port="4"/>
    <edge from-layer="5" from-port="0" to-layer="6" to-port="5"/>
    <edge from-layer="6" from-port="6" to-layer="7" to-port="0"/>
  </edges>
</net>
"""
        xml_path = outdir / f"slat_io_{safe}_fused_resblock.xml"
        with open(str(xml_path), "w") as f:
            f.write(xml)
        print(f"    {safe}: C_in={C_in} C_mid={C_mid} C_out={C_out} "
              f"skip={has_skip} ({n_params} params, {packed.nbytes/1e6:.1f} MB)")

def save_idx_emb(sd, cond_pfx, outdir, name):
    k=cond_pfx+"idx_emb"
    if k in sd:
        emb=sd[k].float().numpy()
        p=outdir/f"{name}_idx_emb.npy"
        np.save(str(p),emb)
        print(f"    Saved {name} idx_emb {emb.shape}")

def save_pointpatch_outer(sd, outdir):
    pfx=COND_PFX_SS+"module_list.2."
    d={"point_proj_weight":sd[pfx+"point_proj.weight"].float().numpy(),
       "point_proj_bias":sd[pfx+"point_proj.bias"].float().numpy(),
       "invalid_xyz_token":sd[pfx+"invalid_xyz_token"].float().numpy()}
    p=outdir/"ss_pointpatch_outer_weights.npz"
    np.savez(str(p),**d)
    print("    Saved PointPatchEmbed outer weights")

def export_pointpatch_outer(sd, outdir):
    """Export PointPatchEmbed outer (xyz projection) as OV IR model."""
    print("  [SS] PointPatchEmbed outer (xyz → 512 projection)...")
    pfx = COND_PFX_SS + "module_list.2."
    w = sd[pfx + "point_proj.weight"].float()   # (512, 3)
    b = sd[pfx + "point_proj.bias"].float()     # (512,)
    inv = sd[pfx + "invalid_xyz_token"].float() # (512,)
    model = PointPatchOuterModel(w, b, inv)
    model.eval()
    N = 256 * 256
    xyz = torch.zeros(N, 3)
    mask = torch.ones(N, 1)
    torch_to_ov(model, (xyz, mask), outdir, "ss_pointpatch_outer")


def export_pointpatch_full(sd, outdir):
    """Export full PointPatchEmbed (outer proj + inner ViT) as one OV IR model."""
    print("  [SS] PointPatchEmbed full (xyz → 1024 window tokens)...")
    pfx = COND_PFX_SS + "module_list.2."
    model = PointPatchFullModel(sd, pfx)
    model.eval()
    xyz = torch.zeros(1, 3, DINO_INPUT_SIZE, DINO_INPUT_SIZE)
    valid = torch.ones(1, 1, DINO_INPUT_SIZE, DINO_INPUT_SIZE)
    torch_to_ov(model, (xyz, valid), outdir, "ss_pointpatch")


def export_ss_embedder_proj_2(sd, outdir):
    """Export EmbedderFuser projection net #2 (pointpatch 512 → 1024)."""
    print("  [SS] EmbedderFuser projection net #2 (pointpatch 512→1024)...")
    pfx = COND_PFX_SS + "projection_nets.2."
    hidden = sd[pfx + "1.w1.weight"].shape[0]   # 2816
    net = ProjectionNet(POINTPATCH_EMBED_DIM, COND_EMBED_DIM, hidden)
    net.load_sd(sd, pfx)
    inp = torch.zeros(1, POINTPATCH_NUM_WINDOWS, POINTPATCH_EMBED_DIM)
    torch_to_ov(net, inp, outdir, "ss_embedder_proj_2")


def export_slat_decoder_gs(dec_gs_ckpt: Path, outdir: Path):
    """Export SLAT Gaussian decoder as a SINGLE Swin-windowed OV model.

    Replaces the previous 14 XMLs (12 per-block + input + output) with one
    `slat_dec_gs.xml`. Swin windowing is done inside the graph via precomputed
    gather/inverse index tensors passed at inference time. Numerically identical
    to the per-window full-attention path (padding keys are masked out).
    """
    print("  [SLAT] Decoder GS (single windowed graph, latent_ch=8 → 448 features)...")
    sd = load_checkpoint(dec_gs_ckpt)
    blocks = [SLATDecoderBlock(SLAT_DEC_MODEL_CHANNELS, SLAT_DEC_NUM_HEADS)
              for _ in range(SLAT_DEC_NUM_BLOCKS)]
    for i, b in enumerate(blocks):
        b.load_sd(sd, f"blocks.{i}.")
        b.eval()
    model = SLATDecoderWindowedModel(blocks,
                                     in_channels=SLAT_DEC_LATENT_CHANNELS,
                                     model_channels=SLAT_DEC_MODEL_CHANNELS,
                                     out_channels=SLAT_DEC_GS_OUT_CHANNELS)
    model.load_sd(sd)
    model.eval()

    # ── Dummy inputs matching the runtime windowing layout ──
    # 4 voxels, 2 windows of max_occ=2 for each parity (arbitrary small example).
    N = 4
    feats = torch.zeros(N, SLAT_DEC_LATENT_CHANNELS)
    coords = torch.zeros(N, 3)
    # even/odd: 2 windows × 2 slots = L=4 (all valid here)
    perm_e = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    inv_e  = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    mask_e = torch.zeros(2, 2)
    perm_o = perm_e.clone(); inv_o = inv_e.clone(); mask_o = mask_e.clone()

    torch_to_ov(
        model,
        (feats, coords, perm_e, inv_e, mask_e, perm_o, inv_o, mask_o),
        outdir, "slat_dec_gs",
    )
    print("    ✓ GS decoder single windowed OV model: slat_dec_gs.xml")


def export_slat_t_emb_all(slat_sd: dict, outdir: Path):
    """Export merged SLAT t_embedder + 4 IO emb_layers as single OV model.

    Input: scalar t
    Outputs: t_emb(1024), emb_in0(256), emb_in1(2048), emb_out0(2048), emb_out1(256)
    """
    print("  [SLAT] Merged t_embedder + IO emb_layers...")
    backbone_pfx = "_base_models.generator.reverse_fn.backbone."

    # Build t_embedder
    pfx = backbone_pfx + "t_embedder."
    t_emb = TimestepEmbedder(SLAT_MODEL_CHANNELS, freq_size=256)
    t_emb.mlp[0].weight.data.copy_(slat_sd[pfx + "mlp.0.weight"])
    t_emb.mlp[0].bias.data.copy_(slat_sd[pfx + "mlp.0.bias"])
    t_emb.mlp[2].weight.data.copy_(slat_sd[pfx + "mlp.2.weight"])
    t_emb.mlp[2].bias.data.copy_(slat_sd[pfx + "mlp.2.bias"])

    # Build 4 emb projections (just Linear, SiLU is in the merged forward)
    block_names = ["input_blocks.0", "input_blocks.1", "out_blocks.0", "out_blocks.1"]
    emb_linears = []
    for bn in block_names:
        bp = backbone_pfx + bn + "."
        w = slat_sd[bp + "emb_layers.1.weight"]
        b = slat_sd[bp + "emb_layers.1.bias"]
        lin = nn.Linear(1024, w.shape[0], bias=True)
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        emb_linears.append(lin)

    model = SLATTimestepEmbedAll(t_emb, *emb_linears)
    model.eval()
    dummy_t = torch.tensor([500.0])
    torch_to_ov(model, (dummy_t,), outdir, "slat_t_emb_all")


def export_slat_io_layers(slat_sd: dict, outdir: Path):
    """Export SLAT input_layer and out_layer as OV models."""
    pfx = "_base_models.generator.reverse_fn.backbone."

    # input_layer: Linear(8 → 128)
    print("  [SLAT] input_layer (8 → 128)...")
    class SLATInputLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 128, bias=True)
        def forward(self, feats):
            return self.lin(feats)
    inp = SLATInputLayer()
    inp.lin.weight.data.copy_(slat_sd[pfx + "input_layer.weight"])
    inp.lin.bias.data.copy_(slat_sd[pfx + "input_layer.bias"])
    inp.eval()
    torch_to_ov(inp, (torch.zeros(64, 8),), outdir, "slat_input_layer")

    # out_layer: LayerNorm + Linear(128 → 8)
    print("  [SLAT] out_layer (LN + 128 → 8)...")
    class SLATOutLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(128, 8, bias=True)
        def forward(self, feats):
            feats = F.layer_norm(feats.float(), [128])
            return self.lin(feats)
    out = SLATOutLayer()
    out.lin.weight.data.copy_(slat_sd[pfx + "out_layer.weight"])
    out.lin.bias.data.copy_(slat_sd[pfx + "out_layer.bias"])
    out.eval()
    torch_to_ov(out, (torch.zeros(64, 128),), outdir, "slat_out_layer")


def export_slat_downsample_upsample(outdir: Path):
    """Export SparseDownsample(2) and SparseUpsample(2) as OV IR models.

    The baseline SLatFlowModel downsamples voxels 2x (64^3 -> 32^3) before the
    24 attention blocks (input_blocks[1], downsample=True) and upsamples back
    after (out_blocks[0], upsample=True). Downsample = average-pool over each
    occupied 2x2x2 cell; Upsample = replicate each pooled feature back to its
    member voxels (paired index map). The grouping topology (coords//2, unique)
    is deterministic and fixed for the whole ODE loop, so it is prepared once
    from the voxel coordinates and passed in as an index tensor — the actual
    feature averaging / gathering runs here inside OpenVINO.
    """
    print("  [SLAT] Downsample / Upsample (avg-pool 2x / replicate) OV models...")

    # ── SparseDownsample: scatter-mean feats[N,C] -> ds[M,C] using idx[N] ──
    class SparseDownsample(nn.Module):
        def forward(self, feats, idx, counts):
            # feats: (N, C) f32; idx: (N,) i64 group id in [0,M); counts: (M,1) f32
            M = counts.shape[0]
            C = feats.shape[1]
            out = torch.zeros(M, C, dtype=feats.dtype)
            out = out.scatter_add(0, idx.unsqueeze(1).expand(-1, C), feats)
            return out / counts

    ds = SparseDownsample().eval()
    feats = torch.zeros(4, 128)
    idx = torch.zeros(4, dtype=torch.int64)
    counts = torch.ones(2, 1)
    torch_to_ov(ds, (feats, idx, counts), outdir, "slat_downsample")

    # ── SparseUpsample: gather ds[M,C] -> feats[N,C] using idx[N] ──
    class SparseUpsample(nn.Module):
        def forward(self, ds_feats, idx):
            # ds_feats: (M, C) f32; idx: (N,) i64 -> each output voxel's group
            return ds_feats.index_select(0, idx)

    us = SparseUpsample().eval()
    ds_feats = torch.zeros(2, 2048)
    idx = torch.zeros(4, dtype=torch.int64)
    torch_to_ov(us, (ds_feats, idx), outdir, "slat_upsample")


def export_slat_decoder_mesh_attn(dec_mesh_ckpt: Path, outdir: Path):
    """Export SLAT mesh decoder transformer blocks (12 blocks → 768-ch hidden features).

    The SparseSubdivideBlock3d upsample layers (sparse conv3d) are handled
    separately via save_slat_decoder_mesh_upsample_weights().
    """
    print("  [SLAT] Decoder Mesh attn blocks (12 blocks, latent_ch=8 → 768 hidden)...")
    sd = load_checkpoint(dec_mesh_ckpt)
    blocks = [SLATDecoderBlock(SLAT_DEC_MODEL_CHANNELS, SLAT_DEC_NUM_HEADS)
              for _ in range(SLAT_DEC_NUM_BLOCKS)]
    for i, b in enumerate(blocks):
        b.load_sd(sd, f"blocks.{i}.")
        b.eval()
    # For mesh decoder: out_lin = identity (upsample blocks come after in inference)
    # We apply layer_norm in SLATDecoderModel.forward() then the "out_lin" here.
    # To output 768-ch features for the upsample blocks, use a simple identity-weight Linear.
    model = SLATDecoderModel(blocks,
                             in_channels=SLAT_DEC_LATENT_CHANNELS,
                             model_channels=SLAT_DEC_MODEL_CHANNELS,
                             out_channels=SLAT_DEC_MODEL_CHANNELS)
    model.in_lin.weight.data.copy_(sd["input_layer.weight"])
    model.in_lin.bias.data.copy_(sd["input_layer.bias"])
    # out_lin = identity (no out_layer in mesh transformer; upsample blocks handle output)
    model.out_lin.weight.data.copy_(torch.eye(SLAT_DEC_MODEL_CHANNELS))
    model.out_lin.bias.data.zero_()
    model.eval()
    N = 64
    feats = torch.zeros(N, SLAT_DEC_LATENT_CHANNELS)
    coords = torch.zeros(N, 3, dtype=torch.long)
    torch_to_ov(model, (feats, coords), outdir, "slat_decoder_mesh_attn")


def save_slat_decoder_mesh_upsample_weights(dec_mesh_ckpt: Path, outdir: Path):
    """Save SparseSubdivideBlock3d weights for use with sparse conv3d extension."""
    print("  [SLAT] Saving mesh decoder upsample weights (sparse conv3d)...")
    sd = load_checkpoint(dec_mesh_ckpt)
    wdir = outdir / "slat_decoder_mesh_upsample_weights"
    wdir.mkdir(parents=True, exist_ok=True)
    manifest = {"upsample_blocks": [], "out_layer": {}}
    for blk_idx in range(2):
        pfx = f"upsample.{blk_idx}."
        blk_manifest = {}
        for key in [k for k in sd if k.startswith(pfx)]:
            rel = key[len(pfx):]
            arr = sd[key].float().numpy()
            fname = key.replace(".", "_") + ".npy"
            np.save(str(wdir / fname), arr)
            blk_manifest[rel] = {"file": fname, "shape": list(arr.shape)}
        manifest["upsample_blocks"].append(blk_manifest)
    # out_layer (Linear, no sparse conv)
    for suf in ["weight", "bias"]:
        k = f"out_layer.{suf}"
        arr = sd[k].float().numpy()
        fname = k.replace(".", "_") + ".npy"
        np.save(str(wdir / fname), arr)
        manifest["out_layer"][suf] = {"file": fname, "shape": list(arr.shape)}
    mpath = outdir / "slat_decoder_mesh_upsample_manifest.json"
    with open(str(mpath), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"    Saved {len([k for b in manifest['upsample_blocks'] for k in b])} "
          f"weight files to {wdir}")


def export_slat_mesh_upsample(dec_mesh_ckpt: Path, outdir: Path):
    """Export the mesh decoder upsample path as a single OV IR model.

    Fuses both SparseSubdivideBlock3d blocks + out_layer SparseLinear into one
    SparseMeshUpsample custom op (FP32, OpenCL).  Emits a single .xml/.bin pair
    (reducing model count vs. the per-weight .npy manifest).  Weights are packed
    into one FP32 buffer in the exact order the C++ op unpacks them.
    """
    print("  [SLAT] Exporting mesh upsample (SparseMeshUpsample custom op)...")
    sd = load_checkpoint(dec_mesh_ckpt)

    MAX_VOXELS = 65536

    def _t(k):
        return sd[k].float().numpy().astype(np.float32)

    packed_parts = []
    dims = []
    for blk_idx in range(2):
        pfx = f"upsample.{blk_idx}."
        act_g = _t(pfx + "act_layers.0.weight")          # (Cin,)
        act_b = _t(pfx + "act_layers.0.bias")
        c0_w  = _t(pfx + "out_layers.0.conv.weight")      # (Cmid,3,3,3,Cin)
        c0_b  = _t(pfx + "out_layers.0.conv.bias")        # (Cmid,)
        g1_w  = _t(pfx + "out_layers.1.weight")           # (Cmid,)
        g1_b  = _t(pfx + "out_layers.1.bias")
        c3_w  = _t(pfx + "out_layers.3.conv.weight")      # (Cout,3,3,3,Cmid)
        c3_b  = _t(pfx + "out_layers.3.conv.bias")        # (Cout,)
        sk_w  = _t(pfx + "skip_connection.conv.weight")   # (Cout,1,1,1,Cin)
        sk_b  = _t(pfx + "skip_connection.conv.bias")     # (Cout,)

        Cmid = c0_w.shape[0]
        Cin  = c0_w.shape[-1]
        Cout = c3_w.shape[0]
        dims += [Cin, Cmid, Cout]

        # Packing order MUST match gpu_mesh_upsample take() in the C++ op:
        packed_parts += [
            act_g.reshape(-1), act_b.reshape(-1),
            c0_w.reshape(-1),  c0_b.reshape(-1),      # [Cmid,27,Cin]
            g1_w.reshape(-1),  g1_b.reshape(-1),
            c3_w.reshape(-1),  c3_b.reshape(-1),      # [Cout,27,Cmid]
            sk_w.reshape(-1),  sk_b.reshape(-1),      # [Cout,Cin] (k1)
        ]

    ol_w = _t("out_layer.weight")   # (101, 96)
    ol_b = _t("out_layer.bias")     # (101,)
    olout, olin = ol_w.shape
    dims += [int(olin), int(olout)]
    packed_parts += [ol_w.reshape(-1), ol_b.reshape(-1)]

    packed = np.concatenate(packed_parts).astype(np.float32)
    n_params = int(packed.size)
    C_in0 = dims[0]

    bin_path = outdir / "slat_mesh_upsample.bin"
    packed.tofile(str(bin_path))

    dims_flat = ",".join(str(int(x)) for x in dims)
    max_out = MAX_VOXELS * 64
    xml = f"""<?xml version="1.0"?>
<net name="slat_mesh_upsample" version="11">
  <layers>
    <layer id="0" name="features" type="Parameter" version="opset1">
      <data shape="-1,{C_in0}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="features">
        <dim>-1</dim><dim>{C_in0}</dim>
      </port></output>
    </layer>
    <layer id="1" name="coords" type="Parameter" version="opset1">
      <data shape="-1,4" element_type="i32"/>
      <output><port id="0" precision="I32" names="coords">
        <dim>-1</dim><dim>4</dim>
      </port></output>
    </layer>
    <layer id="2" name="num_voxels" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_voxels">
        <dim>1</dim>
      </port></output>
    </layer>
    <layer id="3" name="params" type="Const" version="opset1">
      <data element_type="f32" shape="{n_params}" offset="0" size="{n_params * 4}"/>
      <output><port id="0" precision="FP32">
        <dim>{n_params}</dim>
      </port></output>
    </layer>
    <layer id="4" name="mesh_upsample" type="SparseMeshUpsample" version="sam3d">
      <data max_out="{max_out}" dims="{dims_flat}"/>
      <input>
        <port id="0"><dim>-1</dim><dim>{C_in0}</dim></port>
        <port id="1"><dim>-1</dim><dim>4</dim></port>
        <port id="2"><dim>1</dim></port>
        <port id="3"><dim>{n_params}</dim></port>
      </input>
      <output>
        <port id="4" precision="FP32" names="out_features"><dim>-1</dim><dim>{olout}</dim></port>
        <port id="5" precision="I32" names="out_coords"><dim>-1</dim><dim>4</dim></port>
        <port id="6" precision="I32" names="out_num"><dim>1</dim></port>
      </output>
    </layer>
    <layer id="5" name="out_features_result" type="Result" version="opset1">
      <input><port id="0"><dim>-1</dim><dim>{olout}</dim></port></input>
    </layer>
    <layer id="6" name="out_coords_result" type="Result" version="opset1">
      <input><port id="0"><dim>-1</dim><dim>4</dim></port></input>
    </layer>
    <layer id="7" name="out_num_result" type="Result" version="opset1">
      <input><port id="0"><dim>1</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
    <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
    <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
    <edge from-layer="4" from-port="5" to-layer="6" to-port="0"/>
    <edge from-layer="4" from-port="6" to-layer="7" to-port="0"/>
  </edges>
</net>
"""
    xml_path = outdir / "slat_mesh_upsample.xml"
    with open(str(xml_path), "w") as f:
        f.write(xml)
    print(f"    mesh_upsample dims={dims} ({n_params} params, {packed.nbytes/1e6:.1f} MB)")


def export_slat_flexicubes(outdir: Path, res: int = 256):
    """Export the FlexiCubes / SparseFeatures2Mesh extraction as an OV IR model.

    The FlexiCubesExtract op is weightless (all geometry tables are static in the
    C++ op), so the IR is just three Parameters wired into the custom op. An empty
    .bin is written because OV expects the companion weights file to exist.

    Consumed by run_inference_standalone._flexicubes_extract as "slat_flexicubes".
    """
    print("  [SLAT] Exporting FlexiCubes extraction (FlexiCubesExtract custom op)...")

    C_IN = 101  # matches slat_mesh_upsample out_layer output width

    xml = f"""<?xml version="1.0"?>
<net name="slat_flexicubes" version="11">
  <layers>
    <layer id="0" name="features" type="Parameter" version="opset1">
      <data shape="-1,{C_IN}" element_type="f32"/>
      <output><port id="0" precision="FP32" names="features">
        <dim>-1</dim><dim>{C_IN}</dim>
      </port></output>
    </layer>
    <layer id="1" name="coords" type="Parameter" version="opset1">
      <data shape="-1,4" element_type="i32"/>
      <output><port id="0" precision="I32" names="coords">
        <dim>-1</dim><dim>4</dim>
      </port></output>
    </layer>
    <layer id="2" name="num_voxels" type="Parameter" version="opset1">
      <data shape="1" element_type="i32"/>
      <output><port id="0" precision="I32" names="num_voxels">
        <dim>1</dim>
      </port></output>
    </layer>
    <layer id="3" name="flexicubes" type="FlexiCubesExtract" version="sam3d">
      <data res="{res}"/>
      <input>
        <port id="0"><dim>-1</dim><dim>{C_IN}</dim></port>
        <port id="1"><dim>-1</dim><dim>4</dim></port>
        <port id="2"><dim>1</dim></port>
      </input>
      <output>
        <port id="3" precision="FP32" names="vertices"><dim>-1</dim><dim>3</dim></port>
        <port id="4" precision="I32"  names="faces"><dim>-1</dim><dim>3</dim></port>
        <port id="5" precision="FP32" names="colors"><dim>-1</dim><dim>6</dim></port>
        <port id="6" precision="I32"  names="counts"><dim>2</dim></port>
      </output>
    </layer>
    <layer id="4" name="vertices_result" type="Result" version="opset1">
      <input><port id="0"><dim>-1</dim><dim>3</dim></port></input>
    </layer>
    <layer id="5" name="faces_result" type="Result" version="opset1">
      <input><port id="0"><dim>-1</dim><dim>3</dim></port></input>
    </layer>
    <layer id="6" name="colors_result" type="Result" version="opset1">
      <input><port id="0"><dim>-1</dim><dim>6</dim></port></input>
    </layer>
    <layer id="7" name="counts_result" type="Result" version="opset1">
      <input><port id="0"><dim>2</dim></port></input>
    </layer>
  </layers>
  <edges>
    <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
    <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
    <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
    <edge from-layer="3" from-port="5" to-layer="6" to-port="0"/>
    <edge from-layer="3" from-port="6" to-layer="7" to-port="0"/>
  </edges>
</net>
"""
    with open(str(outdir / "slat_flexicubes.xml"), "w") as f:
        f.write(xml)
    # OV looks for a same-named .bin alongside the .xml; the op has no weights.
    np.zeros(0, dtype=np.float32).tofile(str(outdir / "slat_flexicubes.bin"))
    print(f"    flexicubes res={res} (weightless op)")


def export_moge_ov(hf_cache: Optional[str], outdir: Path, resolution_level: int = 9,
                   image_size: int = 518):
    """Export MoGe depth model to OpenVINO IR.

    Args:
        hf_cache: path to HuggingFace cache dir (e.g. /home/rohit/hf_cache).
                  Model.pt should be at hub/models--Ruicheng--moge-vitl/snapshots/*/model.pt
        outdir:   output directory for moge.xml / moge.bin.
        resolution_level: 0-9; 9 = max quality / max tokens.
        image_size: canonical H=W for tracing (must match DINOv2 input size, default 518).
    """
    print("  [MoGe] Exporting MoGe depth model to OV IR...")
    import glob

    # Locate model.pt: first check explicit hf_cache, then ~/.cache/huggingface
    ckpt_path = None
    search_roots = []
    if hf_cache:
        search_roots.append(hf_cache)
    search_roots.append(os.path.expanduser("~/.cache/huggingface"))
    for root in search_roots:
        pattern = os.path.join(root, "hub", "models--Ruicheng--moge-vitl",
                               "snapshots", "*", "model.pt")
        hits = glob.glob(pattern)
        if hits:
            ckpt_path = hits[0]
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "MoGe model.pt not found in HF cache. "
            "Pre-download with:\n"
            "  HF_HUB_DISABLE_XET=1 python3 -c \""
            "from huggingface_hub import snapshot_download; "
            "snapshot_download('Ruicheng/moge-vitl')\"")

    print(f"    Loading from {ckpt_path}")
    from moge.model import import_model_class_by_version
    # Detect v1 vs v2 from checkpoint model_config keys
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_config = raw_ckpt.get("model_config", {})
    version = "v1" if "neck" not in model_config else "v2"
    print(f"    Detected MoGe version: {version}")
    MoGeModel = import_model_class_by_version(version)
    model = MoGeModel.from_pretrained(ckpt_path)
    model.eval()

    min_t, max_t = model.num_tokens_range
    num_tokens = int(min_t + (resolution_level / 9) * (max_t - min_t))
    print(f"    num_tokens={num_tokens} (level={resolution_level})")

    class _MoGeWrapper(nn.Module):
        def __init__(self, inner, nt):
            super().__init__()
            self.inner = inner
            self.nt = nt
        def forward(self, image):       # (1, 3, H, W) → (1,H,W,3), (1,H,W)
            out = self.inner(image, num_tokens=self.nt)
            return out["points"], out["mask"]

    wrapper = _MoGeWrapper(model, num_tokens).eval()
    S = image_size
    dummy = torch.zeros(1, 3, S, S)
    with torch.no_grad():
        ov_model = ov.convert_model(wrapper, example_input=dummy)

    ov_model.inputs[0].get_tensor().set_names({"image"})
    ov_model.outputs[0].get_tensor().set_names({"points"})
    ov_model.outputs[1].get_tensor().set_names({"mask"})

    save_ov(ov_model, outdir, "moge")
    print("    MoGe exported successfully.")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True); ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--output-dir", default="./exported_models")
    ap.add_argument("--skip-dino", action="store_true")
    ap.add_argument("--skip-decoders", action="store_true",
                    help="Skip SLAT decoder export (GS + mesh)")
    ap.add_argument("--skip-moge", action="store_true",
                    help="Skip MoGe OV export (requires moge package)")
    ap.add_argument("--hf-cache", default=None,
                    help="Path to HuggingFace cache dir for MoGe")
    args=ap.parse_args()
    ws=Path(args.workspace)
    ckpt_dir=Path(args.ckpt_dir) if args.ckpt_dir else ws/"checkpoints"
    outdir=Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    print("="*60); print("SAM3D -> OpenVINO Export")
    print(f"  ckpt: {ckpt_dir}"); print(f"  out:  {outdir}"); print("="*60)
    ss_ckpt=ckpt_dir/"ss_generator.ckpt"
    slat_ckpt=ckpt_dir/"slat_generator.ckpt"
    dec_ckpt=ckpt_dir/"ss_decoder.ckpt"
    for p in [ss_ckpt,slat_ckpt,dec_ckpt]:
        if not p.exists(): raise FileNotFoundError(str(p))
    print("\nLoading SS Generator checkpoint...")
    ss_sd=load_checkpoint(ss_ckpt); print(f"  {len(ss_sd)} keys")
    print("Loading SLAT Generator checkpoint...")
    slat_sd=load_checkpoint(slat_ckpt); print(f"  {len(slat_sd)} keys")
    print("\n[SS Generator]")
    if not args.skip_dino:
        export_ss_dino_image(ss_sd, outdir)
        export_ss_dino_mask(ss_sd, outdir)
    export_ss_embedder_projections(ss_sd, outdir)
    export_ss_backbone(ss_sd, outdir)
    export_ss_latent_projections(ss_sd, outdir)
    export_ss_decoder(dec_ckpt, outdir)
    save_idx_emb(ss_sd, COND_PFX_SS, outdir, "ss")
    save_pointpatch_outer(ss_sd, outdir)          # keep legacy .npz for compatibility
    export_pointpatch_outer(ss_sd, outdir)
    export_pointpatch_full(ss_sd, outdir)
    export_ss_embedder_proj_2(ss_sd, outdir)
    print("\n[SLAT Generator]")
    if not args.skip_dino:
        export_slat_dinos(slat_sd, outdir)
    export_slat_embedder_projections(slat_sd, outdir)
    export_slat_attn_blocks(slat_sd, outdir)
    export_slat_sparse_io_weights(slat_sd, outdir)
    export_slat_io_fused_resblock(slat_sd, outdir)
    save_idx_emb(slat_sd, COND_PFX_SLAT, outdir, "slat")
    export_slat_t_emb_all(slat_sd, outdir)
    export_slat_io_layers(slat_sd, outdir)
    export_slat_downsample_upsample(outdir)

    if not args.skip_decoders:
        print("\n[SLAT Decoders]")
        dec_gs_ckpt   = ckpt_dir / "slat_decoder_gs.ckpt"
        dec_mesh_ckpt = ckpt_dir / "slat_decoder_mesh.ckpt"
        if dec_gs_ckpt.exists():
            export_slat_decoder_gs(dec_gs_ckpt, outdir)
        else:
            print(f"  WARNING: {dec_gs_ckpt} not found — skipping GS decoder export")
        if dec_mesh_ckpt.exists():
            export_slat_decoder_mesh_attn(dec_mesh_ckpt, outdir)
            save_slat_decoder_mesh_upsample_weights(dec_mesh_ckpt, outdir)
            export_slat_mesh_upsample(dec_mesh_ckpt, outdir)
            export_slat_flexicubes(outdir)
        else:
            print(f"  WARNING: {dec_mesh_ckpt} not found — skipping mesh decoder export")

    if not args.skip_moge:
        print("\n[MoGe]")
        hf_cache = args.hf_cache
        if hf_cache is None:
            candidate = ws / "checkpoints" / "hf"
            if candidate.exists():
                hf_cache = str(candidate)
        export_moge_ov(hf_cache, outdir)

    # Write runtime config.json consumed by run_inference_standalone.py
    outdir_abs = outdir.resolve()
    ckpt_dir_abs = ckpt_dir.resolve()
    config = {
        "checkpoint_dir": str(ckpt_dir_abs),
        "dino_input_size": DINO_INPUT_SIZE,
        "cond_embed_dim": COND_EMBED_DIM,
        "pointpatch_input_size": POINTPATCH_INPUT_SIZE,
        "pointpatch_num_tokens": POINTPATCH_NUM_WINDOWS,
        "ss_latent_channels": SS_LATENT_IN_CHANNELS["shape"],
        "ss_num_tokens": SS_SHAPE_LATENT_TOKENS,
        "ss_inference_steps": 25,
        "ss_cfg_strength": 7.0,
        "ss_rescale_t": 3.0,
        "ss_time_scale": 1000.0,
        "ss_spatial_res": 16,
        "slat_latent_channels": 8,
        "slat_inference_steps": 25,
        "slat_cfg_strength": 5.0,
        "slat_cfg_interval": [0, 500],
        "slat_rescale_t": 3.0,
        "slat_time_scale": 1000.0,
        "slat_grid_size": 64,
        "slat_mean": [0.12211431, 0.37204156, -1.26521907, -2.05276058, -3.10432536, -0.11294304, -0.85146744, 0.45506954],
        "slat_std":  [2.37326008, 2.13174402, 2.2413953, 2.30589401, 2.1191894, 1.8969511, 2.41684989, 2.08374642],
        "slat_dec_gs_out_channels": SLAT_DEC_GS_OUT_CHANNELS,
        "slat_dec_num_gaussians": 32,
        "slat_dec_voxel_size": 1.5,
        "slat_dec_resolution": 64,
        "slat_dec_gs_lr": {
            "_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0,
            "_scaling": 1.0, "_rotation": 0.1,
        },
        "slat_dec_gs_scaling_bias": 0.004,
        "slat_dec_gs_opacity_bias": 0.1,
        "slat_dec_gs_perturb_offset": True,
        "exported": {
            "ss_dino_image":            str(outdir_abs / "ss_dino_image.xml"),
            "ss_dino_mask":             str(outdir_abs / "ss_dino_mask.xml"),
            **{f"ss_embedder_proj_{i}": str(outdir_abs / f"ss_embedder_proj_{i}.xml") for i in range(2)},
            "ss_backbone":              str(outdir_abs / "ss_backbone.xml"),
            "ss_latent_in_all":         str(outdir_abs / "ss_latent_in_all.xml"),
            "ss_latent_out_all":        str(outdir_abs / "ss_latent_out_all.xml"),
            "ss_decoder":               str(outdir_abs / "ss_decoder.xml"),
            "ss_idx_emb":               str(outdir_abs / "ss_idx_emb.npy"),
            "ss_pointpatch_outer":      str(outdir_abs / "ss_pointpatch_outer.xml"),
            "ss_pointpatch_outer_weights": str(outdir_abs / "ss_pointpatch_outer_weights.npz"),
            "ss_pointpatch":            str(outdir_abs / "ss_pointpatch.xml"),
            "ss_embedder_proj_2":       str(outdir_abs / "ss_embedder_proj_2.xml"),
            "slat_dino_0":              str(outdir_abs / "slat_dino_0.xml"),
            "slat_dino_1":              str(outdir_abs / "slat_dino_1.xml"),
            **{f"slat_embedder_proj_{i}": str(outdir_abs / f"slat_embedder_proj_{i}.xml") for i in range(2)},
            "slat_attn_blocks":         str(outdir_abs / "slat_attn_blocks.xml"),
            "slat_idx_emb":             str(outdir_abs / "slat_idx_emb.npy"),
            "slat_sparse_io_manifest":  str(outdir_abs / "slat_sparse_io_manifest.json"),
            "slat_sparse_io_weights_dir": str(outdir_abs / "slat_sparse_io_weights"),
            "slat_dec_gs":              str(outdir_abs / "slat_dec_gs.xml"),
            "slat_t_emb_all":           str(outdir_abs / "slat_t_emb_all.xml"),
            "slat_input_layer":         str(outdir_abs / "slat_input_layer.xml"),
            "slat_out_layer":           str(outdir_abs / "slat_out_layer.xml"),
            "slat_downsample":          str(outdir_abs / "slat_downsample.xml"),
            "slat_upsample":            str(outdir_abs / "slat_upsample.xml"),
            **{f"slat_io_{bn.replace('.','_')}_fused_resblock": str(outdir_abs / f"slat_io_{bn.replace('.','_')}_fused_resblock.xml")
               for bn in ["input_blocks.0","input_blocks.1","out_blocks.0","out_blocks.1"]},
            "slat_decoder_mesh_attn":   str(outdir_abs / "slat_decoder_mesh_attn.xml"),
            "slat_mesh_upsample":       str(outdir_abs / "slat_mesh_upsample.xml"),
            "slat_flexicubes":          str(outdir_abs / "slat_flexicubes.xml"),
            "slat_decoder_mesh_upsample_manifest": str(
                outdir_abs / "slat_decoder_mesh_upsample_manifest.json"),
            "slat_decoder_mesh_upsample_weights_dir": str(
                outdir_abs / "slat_decoder_mesh_upsample_weights"),
            "moge":                     str(outdir_abs / "moge.xml"),
        }
    }
    cfg_path = outdir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote {cfg_path}")

    print("\n"+"="*60); print("All exports completed!"); print(f"  {outdir}"); print("="*60)

if __name__=="__main__":
    main()
