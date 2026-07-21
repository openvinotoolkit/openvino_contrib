# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
OpenVINO Native Point Cloud Operations using Single-Output GPU Ops

Uses the C++ OpenVINO extension with GPU acceleration via OpenCL kernels.
Supports FP16 computation on Intel GPUs.

Usage:
    from pointcloud_ops_native_gpu import NativePointCloudOpsGPU
    
    ops = NativePointCloudOpsGPU(precision='f16')  # Default FP16
    dists, idx = ops.knn_points(p1, p2, k=12)
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import openvino as ov


class NativePointCloudOpsGPU:
    """OpenVINO native point cloud ops with GPU + FP16 support for performance."""
    
    def __init__(self, precision: str = 'f16'):
        """
        Initialize GPU point cloud operations.
        
        Args:
            precision: 'f16' (default, fast), 'f32' (accurate), or 'default' (GPU decides)
        """
        self.precision = precision
        self.ext_dir = Path(__file__).parent
        self.ext_path = self.ext_dir / 'build' / 'libopenvino_pointcloud_extension.so'
        self.config_path = self.ext_dir / 'pointcloud_ops_gpu_v2.xml'
        
        if not self.ext_path.exists():
            raise FileNotFoundError(f"Extension not found: {self.ext_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"GPU config not found: {self.config_path}")
        
        self.core = ov.Core()
        self.core.add_extension(str(self.ext_path))
        
        # Configure GPU
        gpu_config = {"CONFIG_FILE": str(self.config_path)}
        if precision == 'f16':
            gpu_config["INFERENCE_PRECISION_HINT"] = "f16"
        elif precision == 'f32':
            gpu_config["INFERENCE_PRECISION_HINT"] = "f32"
        # else: let GPU plugin decide (default is f16)
        
        self.core.set_property("GPU", gpu_config)
        
        print(f"[GPU Native] Extension: {self.ext_path.name}")
        print(f"[GPU Native] Config: {self.config_path.name}")
        print(f"[GPU Native] Precision hint: {precision}")
        
        self._models = {}
    
    def _get_model(self, key: str, xml: str) -> ov.CompiledModel:
        """Get or compile model from XML."""
        if key in self._models:
            return self._models[key]
        
        model = self.core.read_model(xml.encode('utf-8'))
        compiled = self.core.compile_model(model, "GPU")
        self._models[key] = compiled
        return compiled
    
    def knn_points(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        k: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-Nearest Neighbors using KNNPointsSingle GPU op.
        
        Returns:
            dists: [B, N1, K] squared distances
            idx: [B, N1, K] indices
        """
        p1 = np.ascontiguousarray(p1, dtype=np.float32)
        p2 = np.ascontiguousarray(p2, dtype=np.float32)
        
        B, N1, _ = p1.shape
        _, N2, _ = p2.shape
        K2 = k * 2
        
        key = f"knn_{B}_{N1}_{N2}_{k}"
        
        xml = f'''<net name="knn" version="11"><layers>
<layer id="0" name="p1" type="Parameter" version="opset1"><data element_type="f32" shape="{B},{N1},3"/><output><port id="0" precision="FP32" names="p1"><dim>{B}</dim><dim>{N1}</dim><dim>3</dim></port></output></layer>
<layer id="1" name="p2" type="Parameter" version="opset1"><data element_type="f32" shape="{B},{N2},3"/><output><port id="0" precision="FP32" names="p2"><dim>{B}</dim><dim>{N2}</dim><dim>3</dim></port></output></layer>
<layer id="2" name="knn" type="KNNPointsSingle" version="hggd"><data k="{k}"/><input><port id="0"><dim>{B}</dim><dim>{N1}</dim><dim>3</dim></port><port id="1"><dim>{B}</dim><dim>{N2}</dim><dim>3</dim></port></input><output><port id="2" precision="FP32"><dim>{B}</dim><dim>{N1}</dim><dim>{K2}</dim></port></output></layer>
<layer id="3" name="result" type="Result" version="opset1"><input><port id="0"><dim>{B}</dim><dim>{N1}</dim><dim>{K2}</dim></port></input></layer>
</layers><edges><edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/><edge from-layer="2" from-port="2" to-layer="3" to-port="0"/></edges></net>'''
        
        model = self._get_model(key, xml)
        result = model({"p1": p1, "p2": p2})[0]
        
        # Split combined output
        dists = result[:, :, :k].copy()
        idx = result[:, :, k:].astype(np.int32).copy()
        
        return dists, idx
    
    def ball_query(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        k: int = 32,
        radius: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ball Query using BallQuerySingle GPU op.
        
        Returns:
            dists: [B, N1, K] squared distances (-1 for invalid)
            idx: [B, N1, K] indices (-1 for invalid)
        """
        p1 = np.ascontiguousarray(p1, dtype=np.float32)
        p2 = np.ascontiguousarray(p2, dtype=np.float32)
        
        B, N1, _ = p1.shape
        _, N2, _ = p2.shape
        K2 = k * 2
        
        key = f"bq_{B}_{N1}_{N2}_{k}_{radius}"
        
        xml = f'''<net name="bq" version="11"><layers>
<layer id="0" name="p1" type="Parameter" version="opset1"><data element_type="f32" shape="{B},{N1},3"/><output><port id="0" precision="FP32" names="p1"><dim>{B}</dim><dim>{N1}</dim><dim>3</dim></port></output></layer>
<layer id="1" name="p2" type="Parameter" version="opset1"><data element_type="f32" shape="{B},{N2},3"/><output><port id="0" precision="FP32" names="p2"><dim>{B}</dim><dim>{N2}</dim><dim>3</dim></port></output></layer>
<layer id="2" name="bq" type="BallQuerySingle" version="hggd"><data k="{k}" radius="{radius}"/><input><port id="0"><dim>{B}</dim><dim>{N1}</dim><dim>3</dim></port><port id="1"><dim>{B}</dim><dim>{N2}</dim><dim>3</dim></port></input><output><port id="2" precision="FP32"><dim>{B}</dim><dim>{N1}</dim><dim>{K2}</dim></port></output></layer>
<layer id="3" name="result" type="Result" version="opset1"><input><port id="0"><dim>{B}</dim><dim>{N1}</dim><dim>{K2}</dim></port></input></layer>
</layers><edges><edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/><edge from-layer="2" from-port="2" to-layer="3" to-port="0"/></edges></net>'''
        
        model = self._get_model(key, xml)
        result = model({"p1": p1, "p2": p2})[0]
        
        dists = result[:, :, :k].copy()
        idx = result[:, :, k:].astype(np.int32).copy()
        
        return dists, idx
    
    @staticmethod
    def _fps_bucket(n: int) -> int:
        """Round N up to next power-of-2 (min 1024) to keep OV model cache small."""
        s = 1024
        while s < n:
            s *= 2
        return s

    def fps(
        self,
        points: np.ndarray,
        k: int = 1024,
        lengths: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Farthest Point Sampling using FPSSingle or FPSWithLengths GPU op.
        
        Pads N to a power-of-2 bucket so the OV GPU model is compiled once
        per bucket instead of once per unique N (avoids ~210ms recompilation).
        
        Args:
            points: [B, N, 3] source points (may be zero-padded)
            k: number of points to sample per batch
            lengths: [B] actual valid length for each batch (optional)
        
        Returns:
            sampled: [B, K, 3] sampled points
            idx: [B, K] sampled indices
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        B, N, _ = points.shape
        
        # Pad N to bucket size so cache key is stable
        N_pad = self._fps_bucket(N)
        if N_pad > N:
            pad = np.zeros((B, N_pad - N, 3), dtype=np.float32)
            points_padded = np.concatenate([points, pad], axis=1)
        else:
            points_padded = points
        
        if lengths is not None:
            # Clamp lengths to N (original, not padded) to avoid out-of-bounds
            lengths = np.ascontiguousarray(lengths, dtype=np.float32)
            lengths = np.minimum(lengths, float(N))
            
            key = f"fps_len_{B}_{N_pad}_{k}"
            
            xml = f'''<net name="fps_len" version="11"><layers>
<layer id="0" name="points" type="Parameter" version="opset1"><data element_type="f32" shape="{B},{N_pad},3"/><output><port id="0" precision="FP32" names="points"><dim>{B}</dim><dim>{N_pad}</dim><dim>3</dim></port></output></layer>
<layer id="1" name="lengths" type="Parameter" version="opset1"><data element_type="f32" shape="{B}"/><output><port id="0" precision="FP32" names="lengths"><dim>{B}</dim></port></output></layer>
<layer id="2" name="fps" type="FPSWithLengths" version="hggd"><data k="{k}"/><input><port id="0"><dim>{B}</dim><dim>{N_pad}</dim><dim>3</dim></port><port id="1"><dim>{B}</dim></port></input><output><port id="2" precision="FP32"><dim>{B}</dim><dim>{k}</dim><dim>4</dim></port></output></layer>
<layer id="3" name="result" type="Result" version="opset1"><input><port id="0"><dim>{B}</dim><dim>{k}</dim><dim>4</dim></port></input></layer>
</layers><edges><edge from-layer="0" from-port="0" to-layer="2" to-port="0"/><edge from-layer="1" from-port="0" to-layer="2" to-port="1"/><edge from-layer="2" from-port="2" to-layer="3" to-port="0"/></edges></net>'''
            
            model = self._get_model(key, xml)
            result = model({"points": points_padded, "lengths": lengths})[0]
        else:
            # Use original FPSSingle kernel (pad N similarly)
            key = f"fps_{B}_{N_pad}_{k}"
            
            xml = f'''<net name="fps" version="11"><layers>
<layer id="0" name="points" type="Parameter" version="opset1"><data element_type="f32" shape="{B},{N_pad},3"/><output><port id="0" precision="FP32" names="points"><dim>{B}</dim><dim>{N_pad}</dim><dim>3</dim></port></output></layer>
<layer id="1" name="fps" type="FPSSingle" version="hggd"><data k="{k}"/><input><port id="0"><dim>{B}</dim><dim>{N_pad}</dim><dim>3</dim></port></input><output><port id="1" precision="FP32"><dim>{B}</dim><dim>{k}</dim><dim>4</dim></port></output></layer>
<layer id="2" name="result" type="Result" version="opset1"><input><port id="0"><dim>{B}</dim><dim>{k}</dim><dim>4</dim></port></input></layer>
</layers><edges><edge from-layer="0" from-port="0" to-layer="1" to-port="0"/><edge from-layer="1" from-port="1" to-layer="2" to-port="0"/></edges></net>'''
            
            model = self._get_model(key, xml)
            result = model({"points": points_padded})[0]
        
        # Output is [B, K, 4] = [x, y, z, idx]
        sampled = result[:, :, :3].copy()
        idx = result[:, :, 3].astype(np.int32).copy()
        
        # Safety: clamp indices to valid range [0, N-1]
        idx = np.clip(idx, 0, N - 1)
        
        return sampled, idx
    
    def gather(
        self,
        features: np.ndarray,
        idx: np.ndarray,
    ) -> np.ndarray:
        """
        Gather features by indices. Uses numpy since this is simple indexing.
        
        Args:
            features: [B, N, D]
            idx: [B, K] or [B, K1, K2]
            
        Returns:
            gathered: [B, K, D] or [B, K1, K2, D]
        """
        B = features.shape[0]
        idx_shape = idx.shape
        D = features.shape[-1]
        
        # Handle -1 indices (invalid)
        valid_idx = np.maximum(idx, 0)
        
        if len(idx_shape) == 2:
            # [B, K] -> [B, K, D]
            gathered = np.zeros((B, idx_shape[1], D), dtype=features.dtype)
            for b in range(B):
                gathered[b] = features[b, valid_idx[b]]
        else:
            # [B, K1, K2] -> [B, K1, K2, D]
            gathered = np.zeros((*idx_shape, D), dtype=features.dtype)
            for b in range(B):
                for i in range(idx_shape[1]):
                    gathered[b, i] = features[b, valid_idx[b, i]]
        
        return gathered


def create_ops(precision: str = 'default') -> NativePointCloudOpsGPU:
    """Factory function to create GPU ops instance."""
    return NativePointCloudOpsGPU(precision=precision)


if __name__ == "__main__":
    import time
    
    ops = NativePointCloudOpsGPU(precision='default')
    
    np.random.seed(42)
    B, N1, N2, K = 1, 1000, 5000, 12
    
    p1 = np.random.randn(B, N1, 3).astype(np.float32)
    p2 = np.random.randn(B, N2, 3).astype(np.float32)
    
    # Warmup
    for _ in range(3):
        ops.knn_points(p1, p2, k=K)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        dists, idx = ops.knn_points(p1, p2, k=K)
    elapsed = (time.time() - start) / 10 * 1000
    
    print(f"\nKNN {N1} x {N2}, K={K}: {elapsed:.2f} ms")
    print(f"Sample dists: {dists[0, 0, :5]}")
    print(f"Sample idx: {idx[0, 0, :5]}")
