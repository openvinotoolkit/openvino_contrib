#!/usr/bin/env python3
"""Run the FlashOCC OpenVINO split pipeline end to end and report latency.

This is a lightweight wrapper around the reusable pipeline pieces in
[run_compare_flashocc_pt_ov.py](run_compare_flashocc_pt_ov.py). It runs only the
OpenVINO path, optionally using the BEV pool extension, and prints per-stage and
end-to-end latency on real nuScenes samples.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from run_compare_flashocc_pt_ov import (
    _BEV_POOL_OPENCL_LOADED,
    _BEVPoolOpenCL,
    _REPO_ROOT,
    _bev_pool_opencl_available,
    BEVPoolOVSO,
    OpenVINOSplitPipeline,
    load_sample_inputs,
    print_latency_throughput,
    print_ov_image_encoder_profile,
)


def _build_bev_pool_backend(args: argparse.Namespace):
    bev_pool_cl = None
    bev_pool_so = None

    ext_so = Path(args.ov_extension_so)
    want_so = (not args.no_so_bev_pool and not args.use_opencl_bev_pool)

    if want_so and ext_so.exists():
        if args.ov_device.upper().startswith('GPU') and not Path(args.ov_gpu_config_xml).exists():
            print(f'  [WARNING] OV extension (.so) found but CONFIG_FILE not found: {args.ov_gpu_config_xml}')
            print('  [WARNING] Falling back to NumPy BEV pool.')
        else:
            print('  Initializing BEV pool via OpenVINO extension (.so) ...')
            bev_pool_so = BEVPoolOVSO(
                extension_so=str(ext_so),
                device=args.ov_device,
                gpu_config_xml=args.ov_gpu_config_xml,
                gpu_precision=args.ov_gpu_precision,
                inference_precision=(args.ov_bevpool_inference_precision or args.ov_inference_precision),
                verbose=True,
            )
    elif want_so and not ext_so.exists():
        print(f'  [INFO] OV extension .so not found at {ext_so}  → using NumPy BEV pool.')
        print('  [INFO] Build with: cd openvino_extensions/bev_pool && mkdir build && cd build && cmake .. && make -j')

    if args.use_opencl_bev_pool and bev_pool_so is None:
        if not _BEV_POOL_OPENCL_LOADED:
            print('  [WARNING] --use-opencl-bev-pool requested but bev_pool_opencl module could not be loaded.')
            print('  [WARNING] Falling back to NumPy BEV pool.')
        elif not _bev_pool_opencl_available():
            print('  [WARNING] --use-opencl-bev-pool requested but no OpenCL devices are available.')
            print('  [WARNING] Falling back to NumPy BEV pool.')
        else:
            print(f'  Initializing OpenCL BEV pool (prefer_gpu={args.opencl_prefer_gpu}) ...')
            bev_pool_cl = _BEVPoolOpenCL(prefer_gpu=args.opencl_prefer_gpu, verbose=True)

    return bev_pool_cl, bev_pool_so


def _load_infos(pkl_path: Path):
    with pkl_path.open('rb') as f:
        data = pickle.load(f)
    return data['infos'] if isinstance(data, dict) else data


def _select_samples(infos: list[dict], start_index: int, num_samples: int) -> tuple[list[int], list[dict]]:
    if start_index < 0 or start_index >= len(infos):
        raise ValueError(f'--start-index out of range: {start_index} for {len(infos)} samples')
    end_index = min(start_index + num_samples, len(infos))
    indices = list(range(start_index, end_index))
    return indices, [infos[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description='FlashOCC end-to-end OpenVINO latency runner')
    parser.add_argument('--model-dir', required=True,
                        help='OpenVINO split model dir (contains *.image_encoder.xml and *.bev_trunk.xml)')
    parser.add_argument('--data-pkl', required=True,
                        help='nuScenes info pkl used to load real samples')
    parser.add_argument('--data-root', required=True,
                        help='nuScenes data root')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of timed samples to run')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations before timing')
    parser.add_argument('--start-index', type=int, default=0,
                        help='First sample index in the pkl to use')
    parser.add_argument('--ov-device', default='CPU',
                        help='OpenVINO target device for image_encoder + bev_trunk (CPU, GPU, NPU, AUTO, ...)')
    parser.add_argument('--ov-gpu-precision', choices=['auto', 'f16', 'f32'], default='f32')
    parser.add_argument('--ov-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Force OV INFERENCE_PRECISION_HINT for both models')
    parser.add_argument('--ov-enc-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for image_encoder only')
    parser.add_argument('--ov-trk-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Per-model OV precision hint for bev_trunk only')
    parser.add_argument('--ov-profile-image-encoder', action='store_true',
                        help='Enable OV PERF_COUNT for image_encoder and print internal profile summary')
    parser.add_argument('--ov-profile-topk', type=int, default=20,
                        help='Top-K nodes to print when --ov-profile-image-encoder is enabled')
    parser.add_argument('--ov-bevpool-inference-precision', choices=['f32', 'f16'], default=None,
                        help='Precision hint for BEV pool .so backend')
    parser.add_argument('--use-opencl-bev-pool', action='store_true',
                        help='Use the OpenCL BEV pool backend instead of NumPy/.so')
    parser.add_argument('--opencl-prefer-gpu', action='store_true', default=True,
                        help='Prefer GPU for OpenCL BEV pool')
    parser.add_argument('--no-so-bev-pool', action='store_true',
                        help='Disable the .so BEV pool backend and fall back to NumPy/OpenCL')
    parser.add_argument('--ov-extension-so',
                        default=str((_REPO_ROOT / 'openvino_extensions' / 'bev_pool' / 'build' / 'libopenvino_bevpool_extension.so').resolve()),
                        help='Path to the BEV pool extension .so')
    parser.add_argument('--ov-gpu-config-xml',
                        default=str((_REPO_ROOT / 'openvino_extensions' / 'gpu_custom_layers.xml').resolve()),
                        help='GPU custom layers xml for the BEV pool extension backend')
    parser.add_argument('--print-per-sample', action='store_true',
                        help='Print latency breakdown for each timed sample')
    args = parser.parse_args()

    if args.use_opencl_bev_pool and args.no_so_bev_pool:
        print('  [INFO] OpenCL backend requested; .so backend disabled explicitly.')

    infos = _load_infos(Path(args.data_pkl))
    sample_indices, selected_infos = _select_samples(infos, args.start_index, args.num_samples)

    print('=' * 72)
    print('  FlashOCC End-to-End OpenVINO Latency')
    print('=' * 72)
    print(f'  Samples          : {len(sample_indices)} (from index {sample_indices[0]} to {sample_indices[-1]})')
    print(f'  Warmup           : {args.warmup}')
    print(f'  OV device        : {args.ov_device}')
    print(f'  Model dir        : {args.model_dir}')

    print('\n  Preloading real nuScenes samples ...')
    data_root = Path(args.data_root)
    samples = [load_sample_inputs(info, data_root) for info in selected_infos]
    print(f'  Loaded {len(samples)} samples')

    bev_pool_cl, bev_pool_so = _build_bev_pool_backend(args)

    ov_pipe = OpenVINOSplitPipeline(
        args.model_dir,
        args.ov_device,
        args.ov_gpu_precision,
        inference_precision=args.ov_inference_precision,
        enc_inference_precision=args.ov_enc_inference_precision,
        trk_inference_precision=args.ov_trk_inference_precision,
        bev_pool_cl=bev_pool_cl,
        bev_pool_so=bev_pool_so,
        profile_image_encoder=args.ov_profile_image_encoder,
    )

    total_iters = args.warmup + len(samples)
    timings_list: list[dict] = []

    print('\n' + '─' * 72)
    print('  Running end-to-end inference')
    print('─' * 72)
    for it in range(total_iters):
        sample_idx = it % len(samples)
        sample = samples[sample_idx]
        _, _, timings, nan_debug, _ = ov_pipe.run(sample, debug_geom=False)

        if it < args.warmup:
            print(f'  Warmup {it + 1:2d}/{args.warmup}: total={timings["total"] * 1000:.0f}ms')
            continue

        timings_list.append(timings)
        logical_idx = sample_indices[sample_idx]
        if args.print_per_sample:
            bev_key = next((k for k in timings.keys() if k.startswith('bev_pool_')), 'bev_pool')
            print(
                f'  Sample {logical_idx:4d}: '
                f'enc={timings["image_encoder"] * 1000:.1f}ms '
                f'{bev_key}={timings[bev_key] * 1000:.1f}ms '
                f'trk={timings["bev_trunk"] * 1000:.1f}ms '
                f'total={timings["total"] * 1000:.1f}ms '
                f'nan[enc={nan_debug["enc_tran_feat"]["nan_count"]},'
                f'bev={nan_debug["bev_feat"]["nan_count"]},'
                f'trk={nan_debug["occ_pred"]["nan_count"]}]'
            )

    print('\n' + '=' * 72)
    print('  LATENCY & THROUGHPUT')
    print('=' * 72)
    print_latency_throughput(timings_list, 'OpenVINO split e2e')
    if args.ov_profile_image_encoder:
        print_ov_image_encoder_profile(ov_pipe.get_image_encoder_profile_summary(topk=args.ov_profile_topk))


if __name__ == '__main__':
    main()
