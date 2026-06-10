#!/usr/bin/env python3
"""
FlashOCC → ONNX → OpenVINO Conversion
=======================================
Exports the FlashOCC model to ONNX with a custom FlashOCCBEVPoolV2 op,
then compiles it with OpenVINO for CPU (or GPU) inference.

Usage:
    source venv_ov2026_ws/bin/activate
    cd <openvino_contrib>/modules/openvino_flashocc
    python convert_to_openvino.py

Prerequisites:
    1. Build the custom OpenVINO extension (one-time):
    cd openvino_extensions/bev_pool
       mkdir -p build && cd build
       cmake .. -DCMAKE_BUILD_TYPE=Release
       make -j$(nproc)

Outputs:
    work_dirs/flashocc-r50-<variant>/openvino/flashocc.onnx
    work_dirs/flashocc-r50-<variant>/openvino/flashocc.xml  (optional IR)
    work_dirs/flashocc-r50-<variant>/openvino/flashocc.bin
"""

import sys
import os
import warnings
import argparse

import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, os.getcwd())

# Register FlashOCC custom modules (BEVDetOCC, heads, ops, etc.)
import mmdet3d_plugin

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_VARIANTS = {
    "m0": {
        "config": "projects/configs/flashocc/flashocc-r50-M0.py",
        "checkpoint": "checkpoints/flashocc-r50.pth",
        "output_dir": "work_dirs/flashocc-r50-m0/openvino",
    },
    "m1": {
        "config": "projects/configs/flashocc/flashocc-r50.py",
        "checkpoint": "checkpoints/flashocc-r50-m1.pth",
        "output_dir": "work_dirs/flashocc-r50-m1/openvino",
    },
}

DEFAULT_VARIANT = "m1"
DEVICE = 'CPU'   # OpenVINO target device

parser = argparse.ArgumentParser(description="FlashOCC ONNX/OpenVINO converter")
parser.add_argument("--model-variant", type=str, default=DEFAULT_VARIANT, choices=["m0", "m1"],
                    help="Model variant: m0 uses flashocc-r50-M0.py + flashocc-r50.pth, m1 uses flashocc-r50.py + flashocc-r50-m1.pth")
parser.add_argument("--device", type=str, default=DEVICE,
                    help='OpenVINO device (e.g. CPU, GPU, AUTO:GPU,CPU, HETERO:GPU,CPU)')
parser.add_argument("--onnx-fallback", action="store_true",
                    help="Use ONNX-native BEV pooling fallback (GPU-friendly, no custom op)")
parser.add_argument("--gpu-native-bevpool", action="store_true",
                    help="Use FlashOCCBEVPoolV2 custom op on GPU with custom OpenCL kernel config")
parser.add_argument("--benchmark-bev-head-only", action="store_true",
                    help="Measure PyTorch FPS for BEV encoder + occupancy head only (excludes image backbone)")
parser.add_argument("--bev-head-runs", type=int, default=50,
                    help="Number of timed runs for BEV encoder + occupancy head benchmark")
parser.add_argument("--benchmark-backbone-only", action="store_true",
                    help="Measure PyTorch FPS for image backbone+neck only (excludes view-transform/BEV/occ head)")
parser.add_argument("--backbone-runs", type=int, default=50,
                    help="Number of timed runs for image backbone+neck benchmark")
parser.add_argument("--export-mode", type=str, default="single", choices=["single", "split", "both"],
                    help="Export mode: single=full pipeline only, split=component models only, both=export both")
args = parser.parse_args()

variant_cfg = MODEL_VARIANTS[args.model_variant]
CONFIG_FILE = variant_cfg["config"]
CHECKPOINT_FILE = variant_cfg["checkpoint"]
OUTPUT_DIR = variant_cfg["output_dir"]
ONNX_FILE = os.path.join(OUTPUT_DIR, 'flashocc.onnx')
IR_XML_FILE = os.path.join(OUTPUT_DIR, 'flashocc.xml')
SPLIT_DIR = os.path.join(OUTPUT_DIR, 'split')

print(f"  Model variant : {args.model_variant}")
print(f"  Config        : {CONFIG_FILE}")
print(f"  Checkpoint    : {CHECKPOINT_FILE}")
print(f"  Output dir    : {OUTPUT_DIR}")

do_full_export = args.export_mode in ("single", "both")
do_split_export = args.export_mode in ("split", "both")

DEVICE = args.device
is_gpu_target = 'GPU' in DEVICE.upper()
if is_gpu_target and not args.onnx_fallback and not args.gpu_native_bevpool:
    print("  ⚠ GPU device selected: enabling ONNX fallback because custom FlashOCCBEVPoolV2 is not supported by Intel GPU plugin")
    args.onnx_fallback = True

if args.onnx_fallback:
    os.environ["FLASHOCC_ONNX_FALLBACK"] = "1"
else:
    os.environ.setdefault("FLASHOCC_ONNX_FALLBACK", "0")


# ── 1. Load model ──────────────────────────────────────────────────────────────
print("[1/4] Loading model...")
from mmcv import Config
from mmdet3d.models import build_model
import mmcv

cfg   = Config.fromfile(CONFIG_FILE)
cfg.model.pretrained = None
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')
state_dict = checkpoint.get('state_dict', checkpoint)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Disable gradient checkpointing — CheckpointFunction is not ONNX-exportable
for m in model.modules():
    if hasattr(m, 'with_cp'):
        m.with_cp = False

print("  ✓ Model loaded")


# ── 2. Build dummy inputs ──────────────────────────────────────────────────────
print("[2/4] Preparing inputs...")

B, N = 1, 6
imgs = torch.randn(B, N, 3, 256, 704, dtype=torch.float32)

# Correct shapes for BEVDet/FlashOCC view transformer
sensor2egos = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, N, 1, 1)
ego2globals = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(B, N, 1, 1)
intrins = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(B, N, 1, 1)   # must be 3x3
post_rots = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(B, N, 1, 1)
post_trans = torch.zeros(B, N, 3, dtype=torch.float32)
bda = torch.eye(3, dtype=torch.float32).view(1, 3, 3).repeat(B, 1, 1)

img_metas = [dict(img_shape=(256, 704, 3)) for _ in range(B)]

if is_gpu_target and hasattr(model, 'img_view_transformer'):
    vt = model.img_view_transformer
    if hasattr(vt, 'pre_compute'):
        vt.accelerate = True
        vt.initial_flag = False
        with torch.no_grad():
            coor = vt.get_ego_coor(sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)
            vt.init_acceleration_v2(coor)
        print("  ✓ Enabled accelerated precomputed BEV indices for GPU export")

# ── Wrapper: logit-level forward ───────────────────────────────────────────────
# class FlashOCCExportWrapper(nn.Module):
#     """
#     Exports FlashOCC to logits [B, 200, 200, 16, 18].
#     The bev_pool_v2 op inside extract_feat appears as FlashOCCBEVPoolV2
#     in the ONNX graph (via the symbolic in bev_pool.py).
#     Camera parameters are typically fixed calibration and fold to constants;
#     ranks/intervals derived from them are embedded as ONNX initializers.
#     """
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, imgs, sensor2egos, ego2globals, intrins,
#                 post_rots, post_trans, bda):
#         img_inputs = [imgs, sensor2egos, ego2globals, intrins,
#                       post_rots, post_trans, bda]
#         img_metas  = [[{'img_shape': (int(imgs.shape[3]),
#                                       int(imgs.shape[4]), 3)}]]
#         img_feats, _, _ = self.model.extract_feat(
#             points=None, img_inputs=img_inputs, img_metas=img_metas[0])
#         x = img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats
#         return self.model.occ_head(x).contiguous()  # [B, 200, 200, 16, 18]
class FlashOCCStaticWrapper(torch.nn.Module):
    def __init__(self, model, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda):
        super().__init__()
        self.model = model
        self.register_buffer("sensor2egos", sensor2egos)
        self.register_buffer("ego2globals", ego2globals)
        self.register_buffer("intrins", intrins)
        self.register_buffer("post_rots", post_rots)
        self.register_buffer("post_trans", post_trans)
        self.register_buffer("bda", bda)

    def forward(self, imgs):
        B = imgs.shape[0]
        img_inputs = [
            imgs,
            self.sensor2egos.expand(B, -1, -1, -1),
            self.ego2globals.expand(B, -1, -1, -1),
            self.intrins.expand(B, -1, -1, -1),
            self.post_rots.expand(B, -1, -1, -1),
            self.post_trans.expand(B, -1, -1),
            self.bda.expand(B, -1, -1),
        ]
        img_metas = [{"img_shape": (int(imgs.shape[3]), int(imgs.shape[4]), 3)} for _ in range(B)]
        img_feats, _, _ = self.model.extract_feat(points=None, img_inputs=img_inputs, img_metas=img_metas)
        x = img_feats[0] if isinstance(img_feats, (list, tuple)) else img_feats
        return self.model.occ_head(x)


class SplitImageEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.D = model.img_view_transformer.D
        self.C = model.img_view_transformer.out_channels

    def forward(self, img):
        x = self.model.img_backbone(img)
        x = self.model.img_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = self.model.img_view_transformer.depth_net(x)
        depth = x[:, :self.D].softmax(dim=1)
        tran_feat = x[:, self.D:self.D + self.C]
        return tran_feat, depth


class SplitBEVTrunkWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bev):
        x = self.model.img_bev_encoder_backbone(bev)
        x = self.model.img_bev_encoder_neck(x)
        return self.model.occ_head(x)

# wrapper = FlashOCCExportWrapper(model)
wrapper = FlashOCCStaticWrapper(model, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda)
wrapper.eval()

# Test run
print("  Testing wrapper...")
with torch.no_grad():
    test_out = wrapper(imgs)   # <-- static wrapper takes only imgs
print(f"  Output shape: {test_out.shape}")
print(f"  Output range: [{test_out.min():.3f}, {test_out.max():.3f}]")


def benchmark_bev_encoder_occ(model, imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda,
                              warmup=5, runs=50):
    import time

    if not hasattr(model, 'image_encoder') or not hasattr(model, 'img_view_transformer') \
       or not hasattr(model, 'bev_encoder') or not hasattr(model, 'occ_head'):
        print("  ⚠ Skipping BEV+Occ benchmark: required modules not found on model")
        return

    with torch.no_grad():
        img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
        prepared = model.prepare_inputs(img_inputs)
        img_feats, _ = model.image_encoder(prepared[0])
        bev_in, _ = model.img_view_transformer([img_feats] + prepared[1:7])

        for _ in range(warmup):
            bev_feat = model.bev_encoder(bev_in)
            _ = model.occ_head(bev_feat)

        t0 = time.time()
        for _ in range(runs):
            bev_feat = model.bev_encoder(bev_in)
            _ = model.occ_head(bev_feat)
        elapsed = (time.time() - t0) / runs

    print("\n  PyTorch BEV encoder + occupancy head (image backbone excluded):")
    print(f"    Average latency : {elapsed*1000:.1f} ms")
    print(f"    Throughput      : {1/elapsed:.2f} FPS")


def benchmark_image_backbone_only(model, imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda,
                                  warmup=5, runs=50):
    import time

    if not hasattr(model, 'prepare_inputs') or not hasattr(model, 'image_encoder'):
        print("  ⚠ Skipping backbone benchmark: required modules not found on model")
        return

    with torch.no_grad():
        img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
        prepared = model.prepare_inputs(img_inputs)
        image_tensor = prepared[0]

        for _ in range(warmup):
            _ = model.image_encoder(image_tensor)

        t0 = time.time()
        for _ in range(runs):
            _ = model.image_encoder(image_tensor)
        elapsed = (time.time() - t0) / runs

    print("\n  PyTorch image backbone+neck only (view-transform/BEV/occ excluded):")
    print(f"    Average latency : {elapsed*1000:.1f} ms")
    print(f"    Throughput      : {1/elapsed:.2f} FPS")


if args.benchmark_backbone_only:
    print(f"\n[2.4/4] Running image backbone+neck benchmark only ({args.backbone_runs} runs timed)...")
    benchmark_image_backbone_only(
        model=model,
        imgs=imgs,
        sensor2egos=sensor2egos,
        ego2globals=ego2globals,
        intrins=intrins,
        post_rots=post_rots,
        post_trans=post_trans,
        bda=bda,
        warmup=5,
        runs=args.backbone_runs,
    )


if args.benchmark_bev_head_only:
    print(f"\n[2.5/4] Running BEV encoder + occupancy head benchmark only ({args.bev_head_runs} runs timed)...")
    benchmark_bev_encoder_occ(
        model=model,
        imgs=imgs,
        sensor2egos=sensor2egos,
        ego2globals=ego2globals,
        intrins=intrins,
        post_rots=post_rots,
        post_trans=post_trans,
        bda=bda,
        warmup=5,
        runs=args.bev_head_runs,
    )


def export_split_models(model, imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda):
    import openvino as ov

    os.makedirs(SPLIT_DIR, exist_ok=True)
    image_encoder_onnx = os.path.join(SPLIT_DIR, "image_encoder.onnx")
    image_encoder_xml = os.path.join(SPLIT_DIR, "image_encoder.xml")
    bev_trunk_onnx = os.path.join(SPLIT_DIR, "bev_trunk.onnx")
    bev_trunk_xml = os.path.join(SPLIT_DIR, "bev_trunk.xml")

    with torch.no_grad():
        img_inputs = [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]
        prepared = model.prepare_inputs(img_inputs)
        img_feats, _ = model.image_encoder(prepared[0])
        bev_in, _ = model.img_view_transformer([img_feats] + prepared[1:7])
        image_encoder_input = prepared[0].reshape(-1, prepared[0].shape[2], prepared[0].shape[3], prepared[0].shape[4])

        image_encoder_wrapper = SplitImageEncoderWrapper(model).eval()
        tran_feat, depth = image_encoder_wrapper(image_encoder_input)

        bev_trunk_wrapper = SplitBEVTrunkWrapper(model).eval()
        occ_logits = bev_trunk_wrapper(bev_in)

    print(f"\n[3a/4] Exporting split model: image encoder → {image_encoder_onnx}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(
            image_encoder_wrapper,
            (image_encoder_input,),
            image_encoder_onnx,
            input_names=["img"],
            output_names=["tran_feat", "depth"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes={
                "img": {0: "num_images"},
                "tran_feat": {0: "num_images"},
                "depth": {0: "num_images"},
            },
        )

    print(f"[3b/4] Exporting split model: bev trunk → {bev_trunk_onnx}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(
            bev_trunk_wrapper,
            (bev_in,),
            bev_trunk_onnx,
            input_names=["bev"],
            output_names=["occ_pred"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes={"bev": {0: "batch"}, "occ_pred": {0: "batch"}},
        )

    core = ov.Core()
    for onnx_path, xml_path in [
        (image_encoder_onnx, image_encoder_xml),
        (bev_trunk_onnx, bev_trunk_xml),
    ]:
        model_ir = core.read_model(onnx_path)
        ov.save_model(model_ir, xml_path)
        print(f"  ✓ IR saved: {xml_path}")

    print(f"  Split output shape checks:")
    print(f"    image_encoder -> tran_feat {tuple(tran_feat.shape)}, depth {tuple(depth.shape)}")
    print(f"    bev_trunk     -> {tuple(occ_logits.shape)}")


# ── 3. Export to ONNX ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

if do_full_export:
    print(f"\n[3/4] Exporting to ONNX → {ONNX_FILE}")

    # GPU plugin is more robust with static input shapes for this model graph.
    export_dynamic_axes = None if is_gpu_target else {"imgs": {0: "batch"}, "occ_logits": {0: "batch"}}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(
            wrapper,
            (imgs,),
            ONNX_FILE,   # <-- use ONNX_FILE
            input_names=["imgs"],
            output_names=["occ_logits"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes=export_dynamic_axes,
        )

    import onnx
    onnx_model = onnx.load(ONNX_FILE)  # <-- use ONNX_FILE
    inputs  = [n.name for n in onnx_model.graph.input]
    outputs = [n.name for n in onnx_model.graph.output]
    nodes   = [n.op_type for n in onnx_model.graph.node]
    custom_ops = [n.op_type for n in onnx_model.graph.node
                  if n.domain in ('flashocc', 'com.microsoft')]

    print(f"  ✓ ONNX saved")
    print(f"    Inputs:     {inputs}")
    print(f"    Outputs:    {outputs}")
    print(f"    Graph nodes: {len(nodes)}")
    print(f"    Custom ops:  {custom_ops}")
else:
    print("\n[3/4] Skipping full pipeline ONNX export (export-mode=split)")

if do_split_export:
    export_split_models(
        model=model,
        imgs=imgs,
        sensor2egos=sensor2egos,
        ego2globals=ego2globals,
        intrins=intrins,
        post_rots=post_rots,
        post_trans=post_trans,
        bda=bda,
    )


# ── 4. Compile with OpenVINO & benchmark ──────────────────────────────────────
print(f"\n[4/4] Compiling with OpenVINO (device={DEVICE})...")

if not do_full_export:
    print("  Skipping full pipeline compile/benchmark because export-mode=split")
    sys.exit(0)

# Custom extension is required whenever ONNX contains flashocc::FlashOCCBEVPoolV2
need_custom_extension = not args.onnx_fallback
import openvino as ov

if need_custom_extension:
    ext_candidates = [
        os.path.join(os.getcwd(), 'openvino_extensions', 'bev_pool', 'build_ws', 'libopenvino_bevpool_extension.so'),
        os.path.join(os.getcwd(), 'openvino_extensions', 'bev_pool', 'build', 'libopenvino_bevpool_extension.so'),
    ]
    ext_so = next((p for p in ext_candidates if os.path.exists(p)), None)

    if not ext_so:
        print("  ✗ FlashOCC extension .so not built yet.")
        print("    Build it with:")
        print("      cd openvino_extensions/bev_pool")
        print("      cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
        print("      # or, when using setup.sh")
        print("      cmake -B build_ws -DCMAKE_BUILD_TYPE=Release && cmake --build build_ws -j")
        sys.exit(1)

    core = ov.Core()
    core.add_extension(ext_so)
    print(f"  ✓ Loaded FlashOCC extension: {ext_so}")
else:
    core = ov.Core()
    print("  Using ONNX fallback mode (no custom extension required)")

try:
    ov_model = core.read_model(ONNX_FILE)

    # FlashOCCBEVPoolV2 now outputs 4D [B,C,Y,X] (Z=1 folded in).
    # Remove the Squeeze(axis=2) node that immediately follows it in the ONNX
    # graph, since there is no longer a Z dimension to squeeze.
    if need_custom_extension:
        for node in ov_model.get_ops():
            if node.get_type_name() == 'FlashOCCBEVPoolV2':
                bevpool_out = node.output(0)
                for consumer_input in list(bevpool_out.get_target_inputs()):
                    squeeze_node = consumer_input.get_node()
                    if squeeze_node.get_type_name() == 'Squeeze':
                        sq_out = squeeze_node.output(0)
                        for sq_consumer in list(sq_out.get_target_inputs()):
                            sq_consumer.replace_source_output(bevpool_out)
                        print("  ✓ Removed Squeeze(axis=2) following FlashOCCBEVPoolV2 (Z=1 folded into op output)")
                break

    # Use CPU when the custom op is present: HETERO:GPU,CPU crashes (SIGSEGV)
    # when the GPU plugin tries to dispatch FlashOCCBEVPoolV2 to CPU fallback.
    compile_target = DEVICE
    if need_custom_extension and 'GPU' in DEVICE.upper():
        compile_target = 'CPU'
        print(f"  Custom op requires CPU evaluate(); compiling on CPU (encoder+bev_pool+trunk)")
    compile_cfg = {'PERFORMANCE_HINT': 'LATENCY'}
    if 'GPU' in DEVICE.upper():
        compile_cfg['INFERENCE_PRECISION_HINT'] = 'f32'
    compiled = core.compile_model(ov_model, compile_target, compile_cfg)
    print("  ✓ Model compiled on", DEVICE)

    # Save IR generated with/without extension for downstream inference scripts.
    ov.save_model(ov_model, IR_XML_FILE)
    print(f"  ✓ IR saved: {IR_XML_FILE}")

    # Benchmark
    import numpy as np, time
    request = compiled.create_infer_request()
    dummy_imgs = np.random.randn(1, 6, 3, 256, 704).astype(np.float32)

    for _ in range(2):      # warmup
        request.infer({'imgs': dummy_imgs})

    N_RUNS = 10
    t0 = time.time()
    for _ in range(N_RUNS):
        request.infer({'imgs': dummy_imgs})
    elapsed = (time.time() - t0) / N_RUNS

    if args.gpu_native_bevpool:
        print(f"\n  OpenVINO {DEVICE} + FlashOCCBEVPoolV2(.so) average latency: {elapsed*1000:.1f} ms  ({1/elapsed:.2f} FPS)")
    else:
        print(f"\n  OpenVINO {DEVICE} average latency: {elapsed*1000:.1f} ms  ({1/elapsed:.2f} FPS)")

except Exception as e:
    print(f"  ✗ OpenVINO compile failed: {e}")
    if need_custom_extension:
        print("  Make sure the FlashOCC extension is built and loadable.")
    else:
        print("  GPU fallback graph compile failed.")
    raise
