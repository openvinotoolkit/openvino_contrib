#!/usr/bin/env python3
"""Standalone OpenCL Conv2D microbench for GFX scheduling experiments.

This is intentionally outside the plugin runtime path. It is used to validate
whether an OpenCL GPU kernel family is worth porting into the shared GFX
MLIR/runtime contracts.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


CL_SUCCESS = 0
CL_DEVICE_TYPE_GPU = 1 << 2
CL_MEM_READ_ONLY = 1 << 2
CL_MEM_WRITE_ONLY = 1 << 1
CL_TRUE = 1
CL_FALSE = 0
CL_QUEUE_PROFILING_ENABLE = 1 << 1
CL_QUEUE_PROPERTIES = 0x1093
CL_PROGRAM_BUILD_LOG = 0x1183

CL_DEVICE_TYPE = 0x1000
CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003
CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005
CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A
CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034
CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A
CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C
CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020
CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021
CL_DEVICE_LOCAL_MEM_TYPE = 0x1022
CL_DEVICE_LOCAL_MEM_SIZE = 0x1023
CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025
CL_DEVICE_QUEUE_PROPERTIES = 0x102A
CL_DEVICE_IMAGE_SUPPORT = 0x1016
CL_DEVICE_NAME = 0x102B
CL_DRIVER_VERSION = 0x102D
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_OPENCL_C_VERSION = 0x103D
CL_DEVICE_NUMERIC_VERSION = 0x105E
CL_DEVICE_OPENCL_C_FEATURES = 0x106F

CL_KERNEL_WORK_GROUP_SIZE = 0x11B0
CL_KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1
CL_KERNEL_LOCAL_MEM_SIZE = 0x11B2
CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3
CL_KERNEL_PRIVATE_MEM_SIZE = 0x11B4

CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283


@dataclass(frozen=True)
class ConvCase:
    name: str
    height: int
    width: int
    input_channels: int
    output_channels: int
    output_height: int
    output_width: int
    stride: int
    pad: int


@dataclass(frozen=True)
class BenchRow:
    case: str
    variant: str
    local_size: str
    median_ms: float
    min_ms: float
    max_abs_diff: float


class CsvRecorder:
    def __init__(self, summary_best: bool, summary_max_abs: float) -> None:
        self.summary_best = summary_best
        self.summary_max_abs = summary_max_abs
        self.rows: list[BenchRow] = []

    @staticmethod
    def header() -> None:
        print("case,variant,local_size,exec_median_ms,exec_min_ms,max_abs_diff", flush=True)

    @staticmethod
    def status(case: str, variant: str, local_size: str, median: str, minimum: str, detail: str) -> None:
        print(f"{case},{variant},{local_size},{median},{minimum},{detail}", flush=True)

    def result(self, case: str, variant: str, local_size: str, times: list[float], max_abs_diff: float) -> None:
        median_ms = statistics.median(times)
        min_ms = min(times)
        self.rows.append(BenchRow(case, variant, local_size, median_ms, min_ms, max_abs_diff))
        print(f"{case},{variant},{local_size},{median_ms:.3f},{min_ms:.3f},{max_abs_diff:.9f}", flush=True)

    def print_summary(self) -> None:
        if not self.summary_best:
            return
        print("summary,case,best_variant,best_local_size,exec_median_ms,exec_min_ms,max_abs_diff", flush=True)
        cases = sorted({row.case for row in self.rows})
        for case in cases:
            valid_rows = [
                row for row in self.rows
                if row.case == case and row.max_abs_diff <= self.summary_max_abs
            ]
            if not valid_rows:
                print(f"summary,{case},none,none,none,none,no_row_with_max_abs_diff<={self.summary_max_abs}", flush=True)
                continue
            best = min(valid_rows, key=lambda row: (row.median_ms, row.min_ms))
            print(
                f"summary,{case},{best.variant},{best.local_size},"
                f"{best.median_ms:.3f},{best.min_ms:.3f},{best.max_abs_diff:.9f}",
                flush=True,
            )

    def best_row_by_case(self) -> dict[str, BenchRow | None]:
        best_rows: dict[str, BenchRow | None] = {}
        for case in sorted({row.case for row in self.rows}):
            valid_rows = [
                row for row in self.rows
                if row.case == case and row.max_abs_diff <= self.summary_max_abs
            ]
            best_rows[case] = None if not valid_rows else min(valid_rows, key=lambda row: (row.median_ms, row.min_ms))
        return best_rows


CASES = {
    "yolo26x_model_1_conv_s2": ConvCase(
        "yolo26x_model_1_conv_s2", 320, 320, 96, 192, 160, 160, 2, 1
    ),
    "yolo26x_model_3_conv_s2": ConvCase(
        "yolo26x_model_3_conv_s2", 160, 160, 384, 384, 80, 80, 2, 1
    ),
    "yolo26x_model_5_conv_s2": ConvCase(
        "yolo26x_model_5_conv_s2", 80, 80, 768, 768, 40, 40, 2, 1
    ),
    "yolo26x_c2_48_48_k3_160": ConvCase(
        "yolo26x_c2_48_48_k3_160", 160, 160, 48, 48, 160, 160, 1, 1
    ),
    "yolo26x_c4_96_96_k3_80": ConvCase(
        "yolo26x_c4_96_96_k3_80", 80, 80, 96, 96, 80, 80, 1, 1
    ),
    "yolo26x_c6_192_192_k3_40": ConvCase(
        "yolo26x_c6_192_192_k3_40", 40, 40, 192, 192, 40, 40, 1, 1
    ),
}


SCALAR_KERNEL = r"""
__kernel void conv_scalar(__global const float* input,
                          __global const float* weights,
                          __global const float* bias,
                          __global float* output,
                          int H, int W, int IC, int OC, int OH, int OW,
                          int stride, int pad) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int oc = get_global_id(2);
  if (x >= OW || y >= OH || oc >= OC) return;

  float acc = bias[oc];
  for (int ci = 0; ci < IC; ++ci) {
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y * stride + kh - pad;
      if (ih < 0 || ih >= H) continue;
      for (int kw = 0; kw < 3; ++kw) {
        int iw = x * stride + kw - pad;
        if (iw < 0 || iw >= W) continue;
        acc += input[(ci * H + ih) * W + iw] *
               weights[((oc * IC + ci) * 3 + kh) * 3 + kw];
      }
    }
  }
  output[(oc * OH + y) * OW + x] = acc;
}
"""


PACKED_KERNEL_TEMPLATE = r"""
#define BLOCK __BLOCK__
#define SPATIAL __SPATIAL__

__kernel void conv_packed(__global const float* input,
                          __global const float* packed_weights,
                          __global const float* bias,
                          __global float* output,
                          int H, int W, int IC, int OC, int OH, int OW,
                          int stride, int pad) {
  int xtile = get_global_id(0) * SPATIAL;
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * BLOCK;
  if (xtile >= OW || y >= OH || oc_base >= OC) return;

  float acc[BLOCK][SPATIAL];
  for (int lane = 0; lane < BLOCK; ++lane) {
    for (int sx = 0; sx < SPATIAL; ++sx) {
      int oc = oc_base + lane;
      acc[lane][sx] = oc < OC ? bias[oc] : 0.0f;
    }
  }

  for (int ci = 0; ci < IC; ++ci) {
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y * stride + kh - pad;
      if (ih < 0 || ih >= H) continue;
      for (int kw = 0; kw < 3; ++kw) {
        int wbase = (((ocb * IC + ci) * 3 + kh) * 3 + kw) * BLOCK;
        float wv[BLOCK];
        for (int lane = 0; lane < BLOCK; ++lane) {
          wv[lane] = packed_weights[wbase + lane];
        }
        for (int sx = 0; sx < SPATIAL; ++sx) {
          int x = xtile + sx;
          if (x >= OW) continue;
          int iw = x * stride + kw - pad;
          if (iw < 0 || iw >= W) continue;
          float in_v = input[(ci * H + ih) * W + iw];
          for (int lane = 0; lane < BLOCK; ++lane) {
            acc[lane][sx] += in_v * wv[lane];
          }
        }
      }
    }
  }

  for (int lane = 0; lane < BLOCK; ++lane) {
    int oc = oc_base + lane;
    if (oc >= OC) continue;
    for (int sx = 0; sx < SPATIAL; ++sx) {
      int x = xtile + sx;
      if (x < OW) {
        output[(oc * OH + y) * OW + x] = acc[lane][sx];
      }
    }
  }
}
"""


OC16_VECTOR_KERNEL = r"""
__kernel void conv_oc16_vec(__global const float* input,
                            __global const float* packed_weights,
                            __global const float* bias,
                            __global float* output,
                            int H, int W, int IC, int OC, int OH, int OW,
                            int stride, int pad) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * 16;
  if (x >= OW || y >= OH || oc_base >= OC) return;

  float16 acc = vload16(0, bias + oc_base);
  for (int ci = 0; ci < IC; ++ci) {
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y * stride + kh - pad;
      if (ih < 0 || ih >= H) continue;
      for (int kw = 0; kw < 3; ++kw) {
        int iw = x * stride + kw - pad;
        if (iw < 0 || iw >= W) continue;
        float in_v = input[(ci * H + ih) * W + iw];
        float16 wv = vload16(0, packed_weights + (((ocb * IC + ci) * 3 + kh) * 3 + kw) * 16);
        acc = mad((float16)(in_v), wv, acc);
      }
    }
  }
  if (oc_base + 15 < OC) {
    __global float* out = output + (oc_base * OH + y) * OW + x;
    out[0 * OH * OW] = acc.s0;
    out[1 * OH * OW] = acc.s1;
    out[2 * OH * OW] = acc.s2;
    out[3 * OH * OW] = acc.s3;
    out[4 * OH * OW] = acc.s4;
    out[5 * OH * OW] = acc.s5;
    out[6 * OH * OW] = acc.s6;
    out[7 * OH * OW] = acc.s7;
    out[8 * OH * OW] = acc.s8;
    out[9 * OH * OW] = acc.s9;
    out[10 * OH * OW] = acc.sa;
    out[11 * OH * OW] = acc.sb;
    out[12 * OH * OW] = acc.sc;
    out[13 * OH * OW] = acc.sd;
    out[14 * OH * OW] = acc.se;
    out[15 * OH * OW] = acc.sf;
  } else {
    float tmp[16];
    vstore16(acc, 0, tmp);
    for (int lane = 0; lane < 16; ++lane) {
      int oc = oc_base + lane;
      if (oc < OC) output[(oc * OH + y) * OW + x] = tmp[lane];
    }
  }
}
"""


OC16_CI4_VECTOR_KERNEL = r"""
__kernel void conv_oc16_ci4_vec(__global const float* input,
                                __global const float* packed_weights,
                                __global const float* bias,
                                __global float* output,
                                int H, int W, int IC, int OC, int OH, int OW,
                                int stride, int pad) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * 16;
  if (x >= OW || y >= OH || oc_base >= OC) return;

  float16 acc = vload16(0, bias + oc_base);
  for (int ci0 = 0; ci0 < IC; ci0 += 4) {
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y * stride + kh - pad;
      if (ih < 0 || ih >= H) continue;
      for (int kw = 0; kw < 3; ++kw) {
        int iw = x * stride + kw - pad;
        if (iw < 0 || iw >= W) continue;
        int in_base = ih * W + iw;
        int w_base = ((ocb * IC + ci0) * 3 + kh) * 3 + kw;
        float in0 = input[((ci0 + 0) * H * W) + in_base];
        acc = mad((float16)(in0), vload16(0, packed_weights + (w_base + 0 * 9) * 16), acc);
        if (ci0 + 1 < IC) {
          float in1 = input[((ci0 + 1) * H * W) + in_base];
          acc = mad((float16)(in1), vload16(0, packed_weights + (w_base + 1 * 9) * 16), acc);
        }
        if (ci0 + 2 < IC) {
          float in2 = input[((ci0 + 2) * H * W) + in_base];
          acc = mad((float16)(in2), vload16(0, packed_weights + (w_base + 2 * 9) * 16), acc);
        }
        if (ci0 + 3 < IC) {
          float in3 = input[((ci0 + 3) * H * W) + in_base];
          acc = mad((float16)(in3), vload16(0, packed_weights + (w_base + 3 * 9) * 16), acc);
        }
      }
    }
  }
  if (oc_base + 15 < OC) {
    __global float* out = output + (oc_base * OH + y) * OW + x;
    out[0 * OH * OW] = acc.s0;
    out[1 * OH * OW] = acc.s1;
    out[2 * OH * OW] = acc.s2;
    out[3 * OH * OW] = acc.s3;
    out[4 * OH * OW] = acc.s4;
    out[5 * OH * OW] = acc.s5;
    out[6 * OH * OW] = acc.s6;
    out[7 * OH * OW] = acc.s7;
    out[8 * OH * OW] = acc.s8;
    out[9 * OH * OW] = acc.s9;
    out[10 * OH * OW] = acc.sa;
    out[11 * OH * OW] = acc.sb;
    out[12 * OH * OW] = acc.sc;
    out[13 * OH * OW] = acc.sd;
    out[14 * OH * OW] = acc.se;
    out[15 * OH * OW] = acc.sf;
  } else {
    float tmp[16];
    vstore16(acc, 0, tmp);
    for (int lane = 0; lane < 16; ++lane) {
      int oc = oc_base + lane;
      if (oc < OC) output[(oc * OH + y) * OW + x] = tmp[lane];
    }
  }
}
"""


OC16_CI4_PTR_KERNEL = r"""
__kernel void conv_oc16_ci4_ptr(__global const float* input,
                                __global const float* packed_weights,
                                __global const float* bias,
                                __global float* output,
                                int H, int W, int IC, int OC, int OH, int OW,
                                int stride, int pad) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * 16;
  if (x >= OW || y >= OH || oc_base >= OC) return;

  const int HW = H * W;
  const int OHW = OH * OW;
  const int xy = y * OW + x;
  const int y0 = y * stride - pad;
  const int x0 = x * stride - pad;
  const int ocb_ic_base = ocb * IC;

  float16 acc = vload16(0, bias + oc_base);
  for (int ci0 = 0; ci0 < IC; ci0 += 4) {
    const int input_base0 = (ci0 + 0) * HW;
    const int input_base1 = (ci0 + 1) * HW;
    const int input_base2 = (ci0 + 2) * HW;
    const int input_base3 = (ci0 + 3) * HW;
    const int w_ci_base = (ocb_ic_base + ci0) * 9 * 16;
    for (int kh = 0; kh < 3; ++kh) {
      int ih = y0 + kh;
      if (ih < 0 || ih >= H) continue;
      const int input_row_base = ih * W;
      const int w_kh_base = w_ci_base + kh * 3 * 16;
      for (int kw = 0; kw < 3; ++kw) {
        int iw = x0 + kw;
        if (iw < 0 || iw >= W) continue;
        const int in_base = input_row_base + iw;
        __global const float* w_ptr = packed_weights + w_kh_base + kw * 16;
        float in0 = input[input_base0 + in_base];
        acc = mad((float16)(in0), vload16(0, w_ptr + 0 * 9 * 16), acc);
        if (ci0 + 1 < IC) {
          float in1 = input[input_base1 + in_base];
          acc = mad((float16)(in1), vload16(0, w_ptr + 1 * 9 * 16), acc);
        }
        if (ci0 + 2 < IC) {
          float in2 = input[input_base2 + in_base];
          acc = mad((float16)(in2), vload16(0, w_ptr + 2 * 9 * 16), acc);
        }
        if (ci0 + 3 < IC) {
          float in3 = input[input_base3 + in_base];
          acc = mad((float16)(in3), vload16(0, w_ptr + 3 * 9 * 16), acc);
        }
      }
    }
  }
  if (oc_base + 15 < OC) {
    __global float* out = output + oc_base * OHW + xy;
    out[0 * OHW] = acc.s0;
    out[1 * OHW] = acc.s1;
    out[2 * OHW] = acc.s2;
    out[3 * OHW] = acc.s3;
    out[4 * OHW] = acc.s4;
    out[5 * OHW] = acc.s5;
    out[6 * OHW] = acc.s6;
    out[7 * OHW] = acc.s7;
    out[8 * OHW] = acc.s8;
    out[9 * OHW] = acc.s9;
    out[10 * OHW] = acc.sa;
    out[11 * OHW] = acc.sb;
    out[12 * OHW] = acc.sc;
    out[13 * OHW] = acc.sd;
    out[14 * OHW] = acc.se;
    out[15 * OHW] = acc.sf;
  } else {
    float tmp[16];
    vstore16(acc, 0, tmp);
    for (int lane = 0; lane < 16; ++lane) {
      int oc = oc_base + lane;
      if (oc < OC) output[oc * OHW + xy] = tmp[lane];
    }
  }
}
"""


INT8_DOT_KERNELS = r"""
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void int8_dot_scalar(__global const int* a,
                              __global const int* b,
                              __global int* out,
                              int count_words) {
  int gid = get_global_id(0);
  if (gid >= count_words) return;
  int av = a[gid];
  int bv = b[gid];
  int acc = 0;
  for (int lane = 0; lane < 4; ++lane) {
    int shift = lane * 8;
    int ax = (av << (24 - shift)) >> 24;
    int bx = (bv << (24 - shift)) >> 24;
    acc += ax * bx;
  }
  out[gid] = acc;
}

__kernel void int8_dot_packed(__global const int* a,
                              __global const int* b,
                              __global int* out,
                              int count_words) {
  int gid = get_global_id(0);
  if (gid >= count_words) return;
#if defined(__opencl_c_integer_dot_product_input_4x8bit_packed)
  out[gid] = dot_4x8packed_ss_int(a[gid], b[gid]);
#else
  out[gid] = 0x7fffffff;
#endif
}
"""


class OpenCL:
    def __init__(self, allow_wall_time: bool = False) -> None:
        self.lib = ctypes.CDLL("libOpenCL.so.1")
        self.cl_platform_id = ctypes.c_void_p
        self.cl_device_id = ctypes.c_void_p
        self.cl_context = ctypes.c_void_p
        self.cl_command_queue = ctypes.c_void_p
        self.cl_mem = ctypes.c_void_p
        self.cl_program = ctypes.c_void_p
        self.cl_kernel = ctypes.c_void_p
        self.cl_event = ctypes.c_void_p
        self.cl_int = ctypes.c_int
        self.cl_uint = ctypes.c_uint
        self.cl_queue_properties = ctypes.c_long
        self.size_t = ctypes.c_size_t
        self.use_event_profiling = True
        self.allow_wall_time = allow_wall_time
        self.has_create_queue_with_properties = False
        self.has_create_queue_with_properties_khr = False
        self.queue_creation_method = "uninitialized"
        self._bind()
        self.device = self._open_gpu_device()

    def _bind(self) -> None:
        c = self
        specs = [
            (
                "clGetPlatformIDs",
                [c.cl_uint, ctypes.POINTER(c.cl_platform_id), ctypes.POINTER(c.cl_uint)],
                None,
            ),
            (
                "clGetDeviceIDs",
                [
                    c.cl_platform_id,
                    ctypes.c_ulong,
                    c.cl_uint,
                    ctypes.POINTER(c.cl_device_id),
                    ctypes.POINTER(c.cl_uint),
                ],
                None,
            ),
            ("clGetExtensionFunctionAddressForPlatform", [c.cl_platform_id, ctypes.c_char_p], ctypes.c_void_p),
            (
                "clGetDeviceInfo",
                [c.cl_device_id, c.cl_uint, c.size_t, ctypes.c_void_p, ctypes.POINTER(c.size_t)],
                None,
            ),
            (
                "clCreateContext",
                [
                    ctypes.c_void_p,
                    c.cl_uint,
                    ctypes.POINTER(c.cl_device_id),
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(c.cl_int),
                ],
                c.cl_context,
            ),
            (
                "clCreateCommandQueue",
                [c.cl_context, c.cl_device_id, ctypes.c_ulong, ctypes.POINTER(c.cl_int)],
                c.cl_command_queue,
            ),
            (
                "clCreateBuffer",
                [c.cl_context, ctypes.c_ulong, c.size_t, ctypes.c_void_p, ctypes.POINTER(c.cl_int)],
                c.cl_mem,
            ),
            (
                "clCreateProgramWithSource",
                [
                    c.cl_context,
                    c.cl_uint,
                    ctypes.POINTER(ctypes.c_char_p),
                    ctypes.POINTER(c.size_t),
                    ctypes.POINTER(c.cl_int),
                ],
                c.cl_program,
            ),
            (
                "clCreateKernel",
                [c.cl_program, ctypes.c_char_p, ctypes.POINTER(c.cl_int)],
                c.cl_kernel,
            ),
            (
                "clEnqueueWriteBuffer",
                [
                    c.cl_command_queue,
                    c.cl_mem,
                    c.cl_uint,
                    c.size_t,
                    c.size_t,
                    ctypes.c_void_p,
                    c.cl_uint,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
                None,
            ),
            (
                "clEnqueueReadBuffer",
                [
                    c.cl_command_queue,
                    c.cl_mem,
                    c.cl_uint,
                    c.size_t,
                    c.size_t,
                    ctypes.c_void_p,
                    c.cl_uint,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
                None,
            ),
            (
                "clBuildProgram",
                [c.cl_program, c.cl_uint, ctypes.POINTER(c.cl_device_id), ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p],
                None,
            ),
            (
                "clGetProgramBuildInfo",
                [c.cl_program, c.cl_device_id, c.cl_uint, c.size_t, ctypes.c_void_p, ctypes.POINTER(c.size_t)],
                None,
            ),
            (
                "clSetKernelArg",
                [c.cl_kernel, c.cl_uint, c.size_t, ctypes.c_void_p],
                None,
            ),
            (
                "clGetKernelWorkGroupInfo",
                [c.cl_kernel, c.cl_device_id, c.cl_uint, c.size_t, ctypes.c_void_p, ctypes.POINTER(c.size_t)],
                c.cl_int,
            ),
            (
                "clEnqueueNDRangeKernel",
                [
                    c.cl_command_queue,
                    c.cl_kernel,
                    c.cl_uint,
                    ctypes.c_void_p,
                    ctypes.POINTER(c.size_t),
                    ctypes.POINTER(c.size_t),
                    c.cl_uint,
                    ctypes.c_void_p,
                    ctypes.POINTER(c.cl_event),
                ],
                None,
            ),
            (
                "clGetEventProfilingInfo",
                [c.cl_event, c.cl_uint, c.size_t, ctypes.c_void_p, ctypes.POINTER(c.size_t)],
                None,
            ),
            ("clReleaseEvent", [c.cl_event], None),
            ("clFinish", [c.cl_command_queue], None),
        ]
        for name, argtypes, restype in specs:
            fn = getattr(self.lib, name)
            fn.argtypes = argtypes
            if restype is not None:
                fn.restype = restype
        if hasattr(self.lib, "clCreateCommandQueueWithProperties"):
            fn = self.lib.clCreateCommandQueueWithProperties
            fn.argtypes = [
                c.cl_context,
                c.cl_device_id,
                ctypes.POINTER(c.cl_queue_properties),
                ctypes.POINTER(c.cl_int),
            ]
            fn.restype = c.cl_command_queue
            self.has_create_queue_with_properties = True
        if hasattr(self.lib, "clCreateCommandQueueWithPropertiesKHR"):
            fn = self.lib.clCreateCommandQueueWithPropertiesKHR
            fn.argtypes = [
                c.cl_context,
                c.cl_device_id,
                ctypes.POINTER(c.cl_queue_properties),
                ctypes.POINTER(c.cl_int),
            ]
            fn.restype = c.cl_command_queue
            self.has_create_queue_with_properties_khr = True

    @staticmethod
    def check(err: int | ctypes.c_int, what: str) -> None:
        value = err.value if isinstance(err, ctypes.c_int) else int(err)
        if value != CL_SUCCESS:
            raise RuntimeError(f"{what} failed err={value}")

    def _open_gpu_device(self) -> tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]:
        count = self.cl_uint()
        self.check(self.lib.clGetPlatformIDs(0, None, ctypes.byref(count)), "clGetPlatformIDs count")
        platforms = (self.cl_platform_id * count.value)()
        self.check(self.lib.clGetPlatformIDs(count.value, platforms, None), "clGetPlatformIDs")
        platform = self.cl_platform_id(platforms[0])

        device_count = self.cl_uint()
        self.check(
            self.lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, None, ctypes.byref(device_count)),
            "clGetDeviceIDs count",
        )
        devices = (self.cl_device_id * device_count.value)()
        self.check(
            self.lib.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_count.value, devices, None),
            "clGetDeviceIDs",
        )
        device = self.cl_device_id(devices[0])

        err = self.cl_int()
        context = self.lib.clCreateContext(None, 1, ctypes.byref(device), None, None, ctypes.byref(err))
        self.check(err, "clCreateContext")
        queue = None
        with_props_err: int | None = None
        with_props_khr_err: int | None = None
        legacy_err: int | None = None
        if self.has_create_queue_with_properties:
            props = (self.cl_queue_properties * 3)(CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0)
            queue = self.lib.clCreateCommandQueueWithProperties(context, device, props, ctypes.byref(err))
            with_props_err = int(err.value)
            if queue is not None and err.value == CL_SUCCESS:
                self.queue_creation_method = "with_properties"
        if queue is None or err.value != CL_SUCCESS:
            khr_fn = None
            if self.has_create_queue_with_properties_khr:
                khr_fn = self.lib.clCreateCommandQueueWithPropertiesKHR
            else:
                fn_ptr = self.lib.clGetExtensionFunctionAddressForPlatform(
                    platform,
                    b"clCreateCommandQueueWithPropertiesKHR",
                )
                if fn_ptr:
                    queue_with_props_type = ctypes.CFUNCTYPE(
                        self.cl_command_queue,
                        self.cl_context,
                        self.cl_device_id,
                        ctypes.POINTER(self.cl_queue_properties),
                        ctypes.POINTER(self.cl_int),
                    )
                    khr_fn = queue_with_props_type(fn_ptr)
                    self.has_create_queue_with_properties_khr = True
        if (queue is None or err.value != CL_SUCCESS) and self.has_create_queue_with_properties_khr:
            props = (self.cl_queue_properties * 3)(CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0)
            queue = khr_fn(context, device, props, ctypes.byref(err))
            with_props_khr_err = int(err.value)
            if queue is not None and err.value == CL_SUCCESS:
                self.queue_creation_method = "with_properties_khr"
        if queue is None or err.value != CL_SUCCESS:
            queue = self.lib.clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, ctypes.byref(err))
            legacy_err = int(err.value)
            if queue is not None and err.value == CL_SUCCESS:
                self.queue_creation_method = "legacy"
        if queue is None or err.value != CL_SUCCESS:
            if not self.allow_wall_time:
                queue_props = self.device_info_ulong_from(device, CL_DEVICE_QUEUE_PROPERTIES)
                raise RuntimeError(
                    "OpenCL event-profiling queue creation failed; "
                    f"clCreateCommandQueueWithProperties err={with_props_err}, "
                    f"clCreateCommandQueueWithPropertiesKHR err={with_props_khr_err}, "
                    f"clCreateCommandQueue err={legacy_err}, "
                    f"CL_DEVICE_QUEUE_PROPERTIES={queue_props}. "
                    "This target does not satisfy the event-profiling measurement gate; "
                    "rerun with --allow-wall-time only for exploratory timing."
                )
            self.use_event_profiling = False
            queue = self.lib.clCreateCommandQueue(context, device, 0, ctypes.byref(err))
            self.queue_creation_method = "legacy_wall_time"
        self.check(err, "clCreateCommandQueue")
        return device, context, queue

    def device_name(self) -> str:
        return self.device_info_string(CL_DEVICE_NAME)

    def device_info_string(self, param: int) -> str:
        device, _, _ = self.device
        size = self.size_t()
        self.check(self.lib.clGetDeviceInfo(device, param, 0, None, ctypes.byref(size)), "clGetDeviceInfo size")
        buf = ctypes.create_string_buffer(size.value)
        self.check(self.lib.clGetDeviceInfo(device, param, size.value, buf, None), "clGetDeviceInfo")
        return buf.value.decode(errors="replace")

    def device_info_string_optional(self, param: int) -> str:
        device, _, _ = self.device
        size = self.size_t()
        err = self.lib.clGetDeviceInfo(device, param, 0, None, ctypes.byref(size))
        if err != CL_SUCCESS:
            return f"unsupported_err_{err}"
        if size.value == 0:
            return ""
        buf = ctypes.create_string_buffer(size.value)
        err = self.lib.clGetDeviceInfo(device, param, size.value, buf, None)
        if err != CL_SUCCESS:
            return f"unsupported_err_{err}"
        return buf.raw[: size.value].hex()

    def device_info_name_versions_optional(self, param: int) -> str:
        device, _, _ = self.device
        size = self.size_t()
        err = self.lib.clGetDeviceInfo(device, param, 0, None, ctypes.byref(size))
        if err != CL_SUCCESS:
            return f"unsupported_err_{err}"
        entry_size = 4 + 64
        if size.value == 0:
            return ""
        if size.value % entry_size != 0:
            return f"unexpected_size_{size.value}"
        buf = ctypes.create_string_buffer(size.value)
        err = self.lib.clGetDeviceInfo(device, param, size.value, buf, None)
        if err != CL_SUCCESS:
            return f"unsupported_err_{err}"
        raw = buf.raw[: size.value]
        features = []
        for offset in range(0, size.value, entry_size):
            version = struct.unpack_from("I", raw, offset)[0]
            name = raw[offset + 4 : offset + entry_size].split(b"\0", 1)[0].decode(errors="replace")
            features.append(f"{name}:{decode_opencl_version(version)}")
        return ";".join(features)

    def device_info_uint_optional(self, param: int) -> int | str:
        device, _, _ = self.device
        value = self.cl_uint()
        err = self.lib.clGetDeviceInfo(device, param, ctypes.sizeof(value), ctypes.byref(value), None)
        if err != CL_SUCCESS:
            return f"unsupported_err_{err}"
        return int(value.value)

    def device_info_uint(self, param: int) -> int:
        device, _, _ = self.device
        value = self.cl_uint()
        self.check(self.lib.clGetDeviceInfo(device, param, ctypes.sizeof(value), ctypes.byref(value), None), "clGetDeviceInfo")
        return int(value.value)

    def device_info_ulong(self, param: int) -> int:
        device, _, _ = self.device
        return self.device_info_ulong_from(device, param)

    def device_info_ulong_from(self, device: ctypes.c_void_p, param: int) -> int:
        value = ctypes.c_ulong()
        self.check(self.lib.clGetDeviceInfo(device, param, ctypes.sizeof(value), ctypes.byref(value), None), "clGetDeviceInfo")
        return int(value.value)

    def device_info_bool(self, param: int) -> bool:
        return self.device_info_uint(param) != CL_FALSE

    def device_info_size_t(self, param: int) -> int:
        device, _, _ = self.device
        value = self.size_t()
        self.check(self.lib.clGetDeviceInfo(device, param, ctypes.sizeof(value), ctypes.byref(value), None), "clGetDeviceInfo")
        return int(value.value)

    def device_info_size_t_list(self, param: int, count: int) -> list[int]:
        device, _, _ = self.device
        values = (self.size_t * count)()
        self.check(self.lib.clGetDeviceInfo(device, param, ctypes.sizeof(values), values, None), "clGetDeviceInfo")
        return [int(values[index]) for index in range(count)]

    def print_device_capabilities(self) -> None:
        dims = self.device_info_uint(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
        work_item_sizes = "x".join(str(value) for value in self.device_info_size_t_list(CL_DEVICE_MAX_WORK_ITEM_SIZES, dims))
        print(f"device,{self.device_name()}", flush=True)
        print(f"capability,device_type,{self.device_info_ulong(CL_DEVICE_TYPE)}", flush=True)
        print(f"capability,device_version,{self.device_info_string(CL_DEVICE_VERSION)}", flush=True)
        numeric_version = self.device_info_uint_optional(CL_DEVICE_NUMERIC_VERSION)
        print(f"capability,numeric_version,{numeric_version}", flush=True)
        if isinstance(numeric_version, int):
            print(f"capability,numeric_version_decoded,{decode_opencl_version(numeric_version)}", flush=True)
        print(f"capability,opencl_c_version,{self.device_info_string(CL_DEVICE_OPENCL_C_VERSION)}", flush=True)
        print(f"capability,driver_version,{self.device_info_string(CL_DRIVER_VERSION)}", flush=True)
        print(f"capability,max_compute_units,{self.device_info_uint(CL_DEVICE_MAX_COMPUTE_UNITS)}", flush=True)
        print(f"capability,max_work_item_dimensions,{dims}", flush=True)
        print(f"capability,max_work_item_sizes,{work_item_sizes}", flush=True)
        print(f"capability,max_work_group_size,{self.device_info_size_t(CL_DEVICE_MAX_WORK_GROUP_SIZE)}", flush=True)
        print(f"capability,preferred_vector_width_float,{self.device_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)}", flush=True)
        print(f"capability,native_vector_width_float,{self.device_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT)}", flush=True)
        print(f"capability,preferred_vector_width_half,{self.device_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF)}", flush=True)
        print(f"capability,native_vector_width_half,{self.device_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF)}", flush=True)
        print(f"capability,local_mem_type,{self.device_info_uint(CL_DEVICE_LOCAL_MEM_TYPE)}", flush=True)
        print(f"capability,local_mem_size,{self.device_info_ulong(CL_DEVICE_LOCAL_MEM_SIZE)}", flush=True)
        print(f"capability,queue_properties,{self.device_info_ulong(CL_DEVICE_QUEUE_PROPERTIES)}", flush=True)
        print(f"capability,has_clCreateCommandQueueWithProperties,{int(self.has_create_queue_with_properties)}", flush=True)
        print(f"capability,has_clCreateCommandQueueWithPropertiesKHR,{int(self.has_create_queue_with_properties_khr)}", flush=True)
        print(f"capability,queue_creation_method,{self.queue_creation_method}", flush=True)
        print(f"capability,event_profiling_enabled,{int(self.use_event_profiling)}", flush=True)
        print(f"capability,profiling_timer_resolution_ns,{self.device_info_size_t(CL_DEVICE_PROFILING_TIMER_RESOLUTION)}", flush=True)
        print(f"capability,max_constant_buffer_size,{self.device_info_ulong(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)}", flush=True)
        print(f"capability,max_constant_args,{self.device_info_uint(CL_DEVICE_MAX_CONSTANT_ARGS)}", flush=True)
        print(f"capability,image_support,{int(self.device_info_bool(CL_DEVICE_IMAGE_SUPPORT))}", flush=True)
        print(f"capability,extensions,{self.device_info_string(CL_DEVICE_EXTENSIONS)}", flush=True)
        print(f"capability,opencl_c_features,{self.device_info_name_versions_optional(CL_DEVICE_OPENCL_C_FEATURES)}", flush=True)
        print(f"capability,opencl_c_features_hex,{self.device_info_string_optional(CL_DEVICE_OPENCL_C_FEATURES)}", flush=True)

    def profile_device_capabilities(self) -> dict[str, object]:
        dims = self.device_info_uint(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
        numeric_version = self.device_info_uint_optional(CL_DEVICE_NUMERIC_VERSION)
        opencl_c_features = self.device_info_name_versions_optional(CL_DEVICE_OPENCL_C_FEATURES)
        return {
            "device_name": self.device_name(),
            "device_type": self.device_info_ulong(CL_DEVICE_TYPE),
            "device_version": self.device_info_string(CL_DEVICE_VERSION),
            "numeric_version": numeric_version,
            "numeric_version_decoded": decode_opencl_version(numeric_version) if isinstance(numeric_version, int) else None,
            "opencl_c_version": self.device_info_string(CL_DEVICE_OPENCL_C_VERSION),
            "driver_version": self.device_info_string(CL_DRIVER_VERSION),
            "max_compute_units": self.device_info_uint(CL_DEVICE_MAX_COMPUTE_UNITS),
            "max_work_item_dimensions": dims,
            "max_work_item_sizes": self.device_info_size_t_list(CL_DEVICE_MAX_WORK_ITEM_SIZES, dims),
            "max_work_group_size": self.device_info_size_t(CL_DEVICE_MAX_WORK_GROUP_SIZE),
            "preferred_vector_width_float": self.device_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT),
            "native_vector_width_float": self.device_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT),
            "preferred_vector_width_half": self.device_info_uint(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF),
            "native_vector_width_half": self.device_info_uint(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF),
            "local_mem_type": self.device_info_uint(CL_DEVICE_LOCAL_MEM_TYPE),
            "local_mem_size": self.device_info_ulong(CL_DEVICE_LOCAL_MEM_SIZE),
            "queue_properties": self.device_info_ulong(CL_DEVICE_QUEUE_PROPERTIES),
            "has_clCreateCommandQueueWithProperties": self.has_create_queue_with_properties,
            "has_clCreateCommandQueueWithPropertiesKHR": self.has_create_queue_with_properties_khr,
            "queue_creation_method": self.queue_creation_method,
            "event_profiling_enabled": self.use_event_profiling,
            "profiling_timer_resolution_ns": self.device_info_size_t(CL_DEVICE_PROFILING_TIMER_RESOLUTION),
            "max_constant_buffer_size": self.device_info_ulong(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE),
            "max_constant_args": self.device_info_uint(CL_DEVICE_MAX_CONSTANT_ARGS),
            "image_support": self.device_info_bool(CL_DEVICE_IMAGE_SUPPORT),
            "extensions": self.device_info_string(CL_DEVICE_EXTENSIONS).split(),
            "opencl_c_features": [] if not opencl_c_features else opencl_c_features.split(";"),
            "opencl_c_features_hex": self.device_info_string_optional(CL_DEVICE_OPENCL_C_FEATURES),
            "timing_kind": "event_profile" if self.use_event_profiling else "host_wall_time",
        }

    def event_profile_ulong(self, event: ctypes.c_void_p, param: int) -> int:
        value = ctypes.c_ulong()
        self.check(
            self.lib.clGetEventProfilingInfo(event, param, ctypes.sizeof(value), ctypes.byref(value), None),
            "clGetEventProfilingInfo",
        )
        return int(value.value)

    def kernel_info_size_t(self, kernel: ctypes.c_void_p, param: int) -> int:
        device, _, _ = self.device
        value = self.size_t()
        self.check(
            self.lib.clGetKernelWorkGroupInfo(kernel, device, param, ctypes.sizeof(value), ctypes.byref(value), None),
            "clGetKernelWorkGroupInfo",
        )
        return int(value.value)

    def kernel_info_ulong(self, kernel: ctypes.c_void_p, param: int) -> int:
        device, _, _ = self.device
        value = ctypes.c_ulong()
        self.check(
            self.lib.clGetKernelWorkGroupInfo(kernel, device, param, ctypes.sizeof(value), ctypes.byref(value), None),
            "clGetKernelWorkGroupInfo",
        )
        return int(value.value)

    def kernel_info_size_t_list(self, kernel: ctypes.c_void_p, param: int, count: int) -> list[int]:
        device, _, _ = self.device
        values = (self.size_t * count)()
        self.check(
            self.lib.clGetKernelWorkGroupInfo(kernel, device, param, ctypes.sizeof(values), values, None),
            "clGetKernelWorkGroupInfo",
        )
        return [int(values[index]) for index in range(count)]

    def print_kernel_capabilities(self, case: ConvCase, variant: str, kernel: ctypes.c_void_p) -> None:
        compile_wg = "x".join(str(value) for value in self.kernel_info_size_t_list(kernel, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 3))
        print(
            f"kernel_info,{case.name},{variant},max_work_group_size,"
            f"{self.kernel_info_size_t(kernel, CL_KERNEL_WORK_GROUP_SIZE)}",
            flush=True,
        )
        print(
            f"kernel_info,{case.name},{variant},preferred_work_group_size_multiple,"
            f"{self.kernel_info_size_t(kernel, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE)}",
            flush=True,
        )
        print(
            f"kernel_info,{case.name},{variant},compile_work_group_size,{compile_wg}",
            flush=True,
        )
        print(
            f"kernel_info,{case.name},{variant},local_mem_size,{self.kernel_info_ulong(kernel, CL_KERNEL_LOCAL_MEM_SIZE)}",
            flush=True,
        )
        print(
            f"kernel_info,{case.name},{variant},private_mem_size,{self.kernel_info_ulong(kernel, CL_KERNEL_PRIVATE_MEM_SIZE)}",
            flush=True,
        )

    def create_buffer(self, flags: int, nbytes: int) -> ctypes.c_void_p:
        _, context, _ = self.device
        err = self.cl_int()
        buffer = self.lib.clCreateBuffer(context, flags, nbytes, None, ctypes.byref(err))
        self.check(err, "clCreateBuffer")
        return self.cl_mem(buffer)

    def write_buffer(self, buffer: ctypes.c_void_p, array: np.ndarray) -> None:
        _, _, queue = self.device
        self.check(
            self.lib.clEnqueueWriteBuffer(
                queue,
                buffer,
                CL_TRUE,
                0,
                array.nbytes,
                array.ctypes.data_as(ctypes.c_void_p),
                0,
                None,
                None,
            ),
            "clEnqueueWriteBuffer",
        )

    def read_buffer(self, buffer: ctypes.c_void_p, array: np.ndarray) -> None:
        _, _, queue = self.device
        self.check(
            self.lib.clEnqueueReadBuffer(
                queue,
                buffer,
                CL_TRUE,
                0,
                array.nbytes,
                array.ctypes.data_as(ctypes.c_void_p),
                0,
                None,
                None,
            ),
            "clEnqueueReadBuffer",
        )

    def build_kernel(self, source: str, name: str, build_options: str = "-cl-std=CL1.2") -> ctypes.c_void_p:
        device, context, _ = self.device
        encoded = source.encode()
        src = ctypes.c_char_p(encoded)
        src_len = self.size_t(len(encoded))
        err = self.cl_int()
        program = self.lib.clCreateProgramWithSource(context, 1, ctypes.byref(src), ctypes.byref(src_len), ctypes.byref(err))
        self.check(err, "clCreateProgramWithSource")
        build_err = self.lib.clBuildProgram(program, 1, ctypes.byref(device), build_options.encode(), None, None)
        if build_err != CL_SUCCESS:
            log_size = self.size_t()
            self.lib.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, None, ctypes.byref(log_size))
            log = ctypes.create_string_buffer(log_size.value)
            self.lib.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size.value, log, None)
            raise RuntimeError(f"OpenCL build failed err={build_err}\n{log.value.decode(errors='replace')}")
        kernel = self.lib.clCreateKernel(program, name.encode(), ctypes.byref(err))
        self.check(err, "clCreateKernel")
        return self.cl_kernel(kernel)

    def set_args(self, kernel: ctypes.c_void_p, args: list[ctypes.c_void_p | ctypes.c_int]) -> None:
        for index, arg in enumerate(args):
            self.check(
                self.lib.clSetKernelArg(kernel, index, ctypes.sizeof(arg), ctypes.byref(arg)),
                f"clSetKernelArg {index}",
            )

    def run_kernel(
        self,
        kernel: ctypes.c_void_p,
        global_size: tuple[int, int, int],
        local_size: tuple[int, int, int] | None,
        iterations: int,
    ) -> list[float]:
        _, _, queue = self.device
        global_arr = (self.size_t * 3)(*global_size)
        local_arr = None if local_size is None else (self.size_t * 3)(*local_size)
        times = []
        for _ in range(iterations):
            event = self.cl_event()
            start = time.perf_counter()
            self.check(
                self.lib.clEnqueueNDRangeKernel(
                    queue,
                    kernel,
                    3,
                    None,
                    global_arr,
                    local_arr,
                    0,
                    None,
                    ctypes.byref(event),
                ),
                "clEnqueueNDRangeKernel",
            )
            self.check(self.lib.clFinish(queue), "clFinish")
            if self.use_event_profiling:
                start_ns = self.event_profile_ulong(event, CL_PROFILING_COMMAND_START)
                end_ns = self.event_profile_ulong(event, CL_PROFILING_COMMAND_END)
                times.append((end_ns - start_ns) * 1.0e-6)
            else:
                times.append((time.perf_counter() - start) * 1000.0)
            if event:
                self.check(self.lib.clReleaseEvent(event), "clReleaseEvent")
        return times


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def local_label(local_size: tuple[int, int, int] | None) -> str:
    if local_size is None:
        return "auto"
    return f"{local_size[0]}x{local_size[1]}x{local_size[2]}"


def parse_local_size(value: str) -> tuple[int, int, int] | None:
    if value == "auto":
        return None
    parts = value.split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected auto or AxBxC local size, got {value!r}")
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer AxBxC local size, got {value!r}") from exc
    if any(part <= 0 for part in parsed):
        raise argparse.ArgumentTypeError(f"local size values must be positive, got {value!r}")
    return parsed


def decode_opencl_version(value: int) -> str:
    major = (value >> 22) & 0x3FF
    minor = (value >> 12) & 0x3FF
    patch = value & 0xFFF
    return f"{major}.{minor}.{patch}"


def make_input(case: ConvCase) -> np.ndarray:
    size = case.input_channels * case.height * case.width
    return ((np.arange(size, dtype=np.float32) % 251) - 125.0) * 0.001


def make_weights(case: ConvCase) -> np.ndarray:
    size = case.output_channels * case.input_channels * 3 * 3
    return ((np.arange(size, dtype=np.float32) % 127) - 63.0) * 0.0001



def pack_weights(case: ConvCase, weights: np.ndarray, block: int) -> np.ndarray:
    oc_blocks = (case.output_channels + block - 1) // block
    packed = np.zeros((oc_blocks, case.input_channels, 3, 3, block), dtype=np.float32)
    original = weights.reshape(case.output_channels, case.input_channels, 3, 3)
    for oc in range(case.output_channels):
        packed[oc // block, :, :, :, oc % block] = original[oc]
    return packed.reshape(-1)


def upload_case(cl: OpenCL, case: ConvCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]:
    input_data = make_input(case)
    weights = make_weights(case)
    bias = np.zeros((case.output_channels,), dtype=np.float32)
    input_buf = cl.create_buffer(CL_MEM_READ_ONLY, input_data.nbytes)
    weight_buf = cl.create_buffer(CL_MEM_READ_ONLY, weights.nbytes)
    bias_buf = cl.create_buffer(CL_MEM_READ_ONLY, bias.nbytes)
    cl.write_buffer(input_buf, input_data)
    cl.write_buffer(weight_buf, weights)
    cl.write_buffer(bias_buf, bias)
    return input_data, weights, bias, input_buf, weight_buf, bias_buf


def make_int8_words(count_words: int, scale: int) -> np.ndarray:
    values = ((np.arange(count_words * 4, dtype=np.int32) * scale + 17) % 255) - 127
    bytes_view = values.astype(np.int8).reshape(count_words, 4)
    return bytes_view.view(np.int32).reshape(count_words)


def run_int8_dot_smoke(cl: OpenCL, count_words: int, iterations: int, print_kernel_info: bool) -> None:
    if iterations <= 0:
        raise ValueError("--iterations must be positive")
    a = make_int8_words(count_words, 3)
    b = make_int8_words(count_words, 5)
    out_scalar = np.empty((count_words,), dtype=np.int32)
    out_packed = np.empty((count_words,), dtype=np.int32)
    a_buf = cl.create_buffer(CL_MEM_READ_ONLY, a.nbytes)
    b_buf = cl.create_buffer(CL_MEM_READ_ONLY, b.nbytes)
    scalar_out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, out_scalar.nbytes)
    packed_out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, out_packed.nbytes)
    cl.write_buffer(a_buf, a)
    cl.write_buffer(b_buf, b)
    scalar = cl.build_kernel(INT8_DOT_KERNELS, "int8_dot_scalar", "-cl-std=CL3.0")
    packed = cl.build_kernel(INT8_DOT_KERNELS, "int8_dot_packed", "-cl-std=CL3.0")
    if print_kernel_info:
        dummy_case = ConvCase("int8_dot", 1, count_words, 1, 1, 1, count_words, 1, 0)
        cl.print_kernel_capabilities(dummy_case, "int8_dot_scalar", scalar)
        cl.print_kernel_capabilities(dummy_case, "int8_dot_packed", packed)
    count_arg = ctypes.c_int(count_words)
    cl.set_args(scalar, [a_buf, b_buf, scalar_out_buf, count_arg])
    cl.set_args(packed, [a_buf, b_buf, packed_out_buf, count_arg])
    local_candidates: tuple[tuple[int, int, int] | None, ...] = (
        None,
        (32, 1, 1),
        (64, 1, 1),
        (128, 1, 1),
    )
    print("case,variant,local_size,exec_median_ms,exec_min_ms,max_abs_diff", flush=True)
    for variant, kernel, output_buf, output in (
        ("int8_dot_scalar", scalar, scalar_out_buf, out_scalar),
        ("int8_dot_packed", packed, packed_out_buf, out_packed),
    ):
        max_work_group_size = cl.kernel_info_size_t(kernel, CL_KERNEL_WORK_GROUP_SIZE)
        for local in local_candidates:
            if local is not None and local_product(local) > max_work_group_size:
                print(
                    f"int8_dot,{variant},{local_label(local)},skip,skip,"
                    f"max_work_group_size={max_work_group_size}",
                    flush=True,
                )
                continue
            global_size = (count_words, 1, 1) if local is None else (round_up(count_words, local[0]), 1, 1)
            try:
                times = cl.run_kernel(kernel, global_size, local, iterations)
                cl.read_buffer(output_buf, output)
                max_abs = 0.0
                if variant == "int8_dot_packed":
                    max_abs = float(np.max(np.abs(out_packed.astype(np.int64) - out_scalar.astype(np.int64))))
                print(
                    f"int8_dot,{variant},{local_label(local)},"
                    f"{statistics.median(times):.3f},{min(times):.3f},{max_abs:.9f}",
                    flush=True,
                )
            except RuntimeError as exc:
                print(
                    f"int8_dot,{variant},{local_label(local)},"
                    f"error,error,{str(exc).replace(',', ';')}",
                    flush=True,
                )


def common_args(
    case: ConvCase,
    input_buf: ctypes.c_void_p,
    weight_buf: ctypes.c_void_p,
    bias_buf: ctypes.c_void_p,
    output_buf: ctypes.c_void_p,
) -> list[ctypes.c_void_p | ctypes.c_int]:
    return [
        input_buf,
        weight_buf,
        bias_buf,
        output_buf,
        ctypes.c_int(case.height),
        ctypes.c_int(case.width),
        ctypes.c_int(case.input_channels),
        ctypes.c_int(case.output_channels),
        ctypes.c_int(case.output_height),
        ctypes.c_int(case.output_width),
        ctypes.c_int(case.stride),
        ctypes.c_int(case.pad),
    ]


def local_product(local_size: tuple[int, int, int] | None) -> int:
    if local_size is None:
        return 1
    return local_size[0] * local_size[1] * local_size[2]


def make_oc16_ci4_ptr_source(
    kernel_name: str,
    attrs: str = "",
    prefetch_weights: bool = False,
    prefetch_inputs: bool = False,
    unroll_ci: int | None = None,
    unroll_kh: int | None = None,
    unroll_kw: int | None = None,
) -> str:
    decl = "__kernel void conv_oc16_ci4_ptr"
    replacement = f"{attrs}\n__kernel void {kernel_name}" if attrs else f"__kernel void {kernel_name}"
    source = OC16_CI4_PTR_KERNEL.replace(decl, replacement, 1)
    if unroll_ci is not None:
        source = source.replace(
            "  for (int ci0 = 0; ci0 < IC; ci0 += 4) {",
            f"  __attribute__((opencl_unroll_hint({unroll_ci})))\n"
            "  for (int ci0 = 0; ci0 < IC; ci0 += 4) {",
            1,
        )
    if unroll_kh is not None:
        source = source.replace(
            "    for (int kh = 0; kh < 3; ++kh) {",
            f"    __attribute__((opencl_unroll_hint({unroll_kh})))\n"
            "    for (int kh = 0; kh < 3; ++kh) {",
            1,
        )
    if unroll_kw is not None:
        source = source.replace(
            "      for (int kw = 0; kw < 3; ++kw) {",
            f"      __attribute__((opencl_unroll_hint({unroll_kw})))\n"
            "      for (int kw = 0; kw < 3; ++kw) {",
            1,
        )
    if prefetch_weights:
        needle = "__global const float* w_ptr = packed_weights + w_kh_base + kw * 16;\n"
        replacement = (
            needle
            + "        prefetch(w_ptr + 0 * 9 * 16, 16);\n"
            + "        if (ci0 + 1 < IC) prefetch(w_ptr + 1 * 9 * 16, 16);\n"
            + "        if (ci0 + 2 < IC) prefetch(w_ptr + 2 * 9 * 16, 16);\n"
            + "        if (ci0 + 3 < IC) prefetch(w_ptr + 3 * 9 * 16, 16);\n"
        )
        source = source.replace(needle, replacement, 1)
    if prefetch_inputs:
        needle = "        float in0 = input[input_base0 + in_base];\n"
        replacement = (
            "        prefetch(input + input_base0 + in_base, 1);\n"
            + "        if (ci0 + 1 < IC) prefetch(input + input_base1 + in_base, 1);\n"
            + "        if (ci0 + 2 < IC) prefetch(input + input_base2 + in_base, 1);\n"
            + "        if (ci0 + 3 < IC) prefetch(input + input_base3 + in_base, 1);\n"
            + needle
        )
        source = source.replace(needle, replacement, 1)
    return source


def vector_component_name(index: int) -> str:
    return "0123456789abcdef"[index]


def make_ci4_ptr_block_source(kernel_name: str, block: int) -> str:
    if block not in (4, 8, 16):
        raise ValueError(f"unsupported block size: {block}")
    vector_type = f"float{block}"
    vload = f"vload{block}"
    vstore = f"vstore{block}"
    full_store_lines = "\n".join(
        f"    out[{lane} * OHW] = acc.s{vector_component_name(lane)};" for lane in range(block)
    )
    return f"""
__kernel void {kernel_name}(__global const float* input,
                            __global const float* packed_weights,
                            __global const float* bias,
                            __global float* output,
                            int H, int W, int IC, int OC, int OH, int OW,
                            int stride, int pad) {{
  int x = get_global_id(0);
  int y = get_global_id(1);
  int ocb = get_global_id(2);
  int oc_base = ocb * {block};
  if (x >= OW || y >= OH || oc_base >= OC) return;

  const int HW = H * W;
  const int OHW = OH * OW;
  const int xy = y * OW + x;
  const int y0 = y * stride - pad;
  const int x0 = x * stride - pad;
  const int ocb_ic_base = ocb * IC;

  {vector_type} acc = {vload}(0, bias + oc_base);
  for (int ci0 = 0; ci0 < IC; ci0 += 4) {{
    const int input_base0 = (ci0 + 0) * HW;
    const int input_base1 = (ci0 + 1) * HW;
    const int input_base2 = (ci0 + 2) * HW;
    const int input_base3 = (ci0 + 3) * HW;
    const int w_ci_base = (ocb_ic_base + ci0) * 9 * {block};
    for (int kh = 0; kh < 3; ++kh) {{
      int ih = y0 + kh;
      if (ih < 0 || ih >= H) continue;
      const int input_row_base = ih * W;
      const int w_kh_base = w_ci_base + kh * 3 * {block};
      for (int kw = 0; kw < 3; ++kw) {{
        int iw = x0 + kw;
        if (iw < 0 || iw >= W) continue;
        const int in_base = input_row_base + iw;
        __global const float* w_ptr = packed_weights + w_kh_base + kw * {block};
        float in0 = input[input_base0 + in_base];
        acc = mad(({vector_type})(in0), {vload}(0, w_ptr + 0 * 9 * {block}), acc);
        if (ci0 + 1 < IC) {{
          float in1 = input[input_base1 + in_base];
          acc = mad(({vector_type})(in1), {vload}(0, w_ptr + 1 * 9 * {block}), acc);
        }}
        if (ci0 + 2 < IC) {{
          float in2 = input[input_base2 + in_base];
          acc = mad(({vector_type})(in2), {vload}(0, w_ptr + 2 * 9 * {block}), acc);
        }}
        if (ci0 + 3 < IC) {{
          float in3 = input[input_base3 + in_base];
          acc = mad(({vector_type})(in3), {vload}(0, w_ptr + 3 * 9 * {block}), acc);
        }}
      }}
    }}
  }}
  if (oc_base + {block - 1} < OC) {{
    __global float* out = output + oc_base * OHW + xy;
{full_store_lines}
  }} else {{
    float tmp[{block}];
    {vstore}(acc, 0, tmp);
    for (int lane = 0; lane < {block}; ++lane) {{
      int oc = oc_base + lane;
      if (oc < OC) output[oc * OHW + xy] = tmp[lane];
    }}
  }}
}}
"""


def run_oc16_ci4_ptr_autotune(
    cl: OpenCL,
    case: ConvCase,
    iterations: int,
    print_kernel_info: bool,
    candidate_filter: set[str] | None = None,
    local_filter: set[tuple[int, int, int] | None] | None = None,
    recorder: CsvRecorder | None = None,
) -> None:
    _, weights, _, input_buf, _, bias_buf = upload_case(cl, case)
    output_size = case.output_channels * case.output_height * case.output_width
    output = np.empty((output_size,), dtype=np.float32)
    reference = None

    packed = pack_weights(case, weights, 16)
    packed_buf = cl.create_buffer(CL_MEM_READ_ONLY, packed.nbytes)
    cl.write_buffer(packed_buf, packed)

    candidates: tuple[
        tuple[
            str,
            str,
            bool,
            bool,
            tuple[int, int, int] | None,
            int | None,
            int | None,
            int | None,
            str,
        ],
        ...,
    ] = (
        ("base", "", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("mad_enable", "", False, False, None, None, None, None, "-cl-std=CL1.2 -cl-mad-enable"),
        ("no_signed_zeros", "", False, False, None, None, None, None, "-cl-std=CL1.2 -cl-no-signed-zeros"),
        ("finite_math", "", False, False, None, None, None, None, "-cl-std=CL1.2 -cl-finite-math-only"),
        ("fast_relaxed", "", False, False, None, None, None, None, "-cl-std=CL1.2 -cl-fast-relaxed-math"),
        ("vec_hint_float4", "__attribute__((vec_type_hint(float4)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("vec_hint_float8", "__attribute__((vec_type_hint(float8)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("vec_hint_float16", "__attribute__((vec_type_hint(float16)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("wg_hint_32x1x1", "__attribute__((work_group_size_hint(32, 1, 1)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("wg_hint_16x2x1", "__attribute__((work_group_size_hint(16, 2, 1)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("wg_hint_16x1x2", "__attribute__((work_group_size_hint(16, 1, 2)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("wg_hint_8x4x1", "__attribute__((work_group_size_hint(8, 4, 1)))", False, False, None, None, None, None, "-cl-std=CL1.2"),
        ("reqd_32x1x1", "__attribute__((reqd_work_group_size(32, 1, 1)))", False, False, (32, 1, 1), None, None, None, "-cl-std=CL1.2"),
        ("reqd_16x2x1", "__attribute__((reqd_work_group_size(16, 2, 1)))", False, False, (16, 2, 1), None, None, None, "-cl-std=CL1.2"),
        ("reqd_16x1x2", "__attribute__((reqd_work_group_size(16, 1, 2)))", False, False, (16, 1, 2), None, None, None, "-cl-std=CL1.2"),
        ("reqd_8x4x1", "__attribute__((reqd_work_group_size(8, 4, 1)))", False, False, (8, 4, 1), None, None, None, "-cl-std=CL1.2"),
        ("unroll_k3", "", False, False, None, None, 3, 3, "-cl-std=CL1.2"),
        ("unroll_ci4_k3", "", False, False, None, 4, 3, 3, "-cl-std=CL1.2"),
        ("prefetch_w", "", True, False, None, None, None, None, "-cl-std=CL1.2"),
        ("prefetch_in", "", False, True, None, None, None, None, "-cl-std=CL1.2"),
        ("prefetch_w_in", "", True, True, None, None, None, None, "-cl-std=CL1.2"),
    )
    local_candidates: tuple[tuple[int, int, int] | None, ...] = (
        None,
        (32, 1, 1),
        (16, 2, 1),
        (8, 4, 1),
        (16, 1, 2),
        (8, 2, 2),
        (8, 1, 4),
    )

    if recorder is None:
        recorder = CsvRecorder(summary_best=False, summary_max_abs=0.0)
        recorder.header()
    for (
        variant_name,
        attrs,
        prefetch_weights,
        prefetch_inputs,
        required_local,
        unroll_ci,
        unroll_kh,
        unroll_kw,
        build_options,
    ) in candidates:
        if candidate_filter is not None and variant_name not in candidate_filter:
            continue
        kernel_name = f"conv_oc16_ci4_ptr_autotune_{variant_name}"
        source = make_oc16_ci4_ptr_source(
            kernel_name,
            attrs,
            prefetch_weights,
            prefetch_inputs,
            unroll_ci,
            unroll_kh,
            unroll_kw,
        )
        try:
            kernel = cl.build_kernel(source, kernel_name, build_options)
        except RuntimeError as exc:
            recorder.status(case.name, variant_name, "build", "build_error", "build_error", str(exc).replace(",", ";"))
            continue
        if print_kernel_info:
            cl.print_kernel_capabilities(case, variant_name, kernel)
        max_work_group_size = cl.kernel_info_size_t(kernel, CL_KERNEL_WORK_GROUP_SIZE)
        active_locals = (required_local,) if required_local is not None else local_candidates
        for local in active_locals:
            if local_filter is not None and local not in local_filter:
                continue
            if local is not None and local_product(local) > max_work_group_size:
                recorder.status(
                    case.name,
                    variant_name,
                    local_label(local),
                    "skip",
                    "skip",
                    f"max_work_group_size={max_work_group_size}",
                )
                continue
            out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
            cl.set_args(kernel, common_args(case, input_buf, packed_buf, bias_buf, out_buf))
            if local is None:
                global_size = (
                    case.output_width,
                    case.output_height,
                    (case.output_channels + 15) // 16,
                )
            else:
                global_size = (
                    round_up(case.output_width, local[0]),
                    round_up(case.output_height, local[1]),
                    round_up((case.output_channels + 15) // 16, local[2]),
                )
            try:
                times = cl.run_kernel(kernel, global_size, local, iterations)
                cl.read_buffer(out_buf, output)
                if reference is None:
                    reference = output.copy()
                max_abs = float(np.max(np.abs(output - reference)))
                recorder.result(case.name, variant_name, local_label(local), times, max_abs)
            except RuntimeError as exc:
                recorder.status(case.name, variant_name, local_label(local), "error", "error", str(exc).replace(",", ";"))


def run_ci4_ptr_block_autotune(cl: OpenCL, case: ConvCase, iterations: int, print_kernel_info: bool) -> None:
    _, weights, _, input_buf, _, bias_buf = upload_case(cl, case)
    output_size = case.output_channels * case.output_height * case.output_width
    output = np.empty((output_size,), dtype=np.float32)
    reference = None

    local_candidates: tuple[tuple[int, int, int] | None, ...] = (
        None,
        (32, 1, 1),
    )

    print("case,variant,local_size,exec_median_ms,exec_min_ms,max_abs_diff", flush=True)
    for block in (16, 8, 4):
        packed = pack_weights(case, weights, block)
        packed_buf = cl.create_buffer(CL_MEM_READ_ONLY, packed.nbytes)
        cl.write_buffer(packed_buf, packed)
        variant = f"oc{block}_ci4_ptr_block"
        kernel_name = f"conv_{variant}"
        source = make_ci4_ptr_block_source(kernel_name, block)
        try:
            kernel = cl.build_kernel(source, kernel_name)
        except RuntimeError as exc:
            print(
                f"{case.name},{variant},build,build_error,build_error,"
                f"{str(exc).replace(',', ';')}",
                flush=True,
            )
            continue
        if print_kernel_info:
            cl.print_kernel_capabilities(case, variant, kernel)
        max_work_group_size = cl.kernel_info_size_t(kernel, CL_KERNEL_WORK_GROUP_SIZE)
        for local in local_candidates:
            if local is not None and local_product(local) > max_work_group_size:
                print(
                    f"{case.name},{variant},{local_label(local)},skip,skip,"
                    f"max_work_group_size={max_work_group_size}",
                    flush=True,
                )
                continue
            out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
            cl.set_args(kernel, common_args(case, input_buf, packed_buf, bias_buf, out_buf))
            if local is None:
                global_size = (
                    case.output_width,
                    case.output_height,
                    (case.output_channels + block - 1) // block,
                )
            else:
                global_size = (
                    round_up(case.output_width, local[0]),
                    round_up(case.output_height, local[1]),
                    round_up((case.output_channels + block - 1) // block, local[2]),
                )
            try:
                times = cl.run_kernel(kernel, global_size, local, iterations)
                cl.read_buffer(out_buf, output)
                if reference is None:
                    reference = output.copy()
                max_abs = float(np.max(np.abs(output - reference)))
                print(
                    f"{case.name},{variant},{local_label(local)},"
                    f"{statistics.median(times):.3f},{min(times):.3f},{max_abs:.9f}",
                    flush=True,
                )
            except RuntimeError as exc:
                print(
                    f"{case.name},{variant},{local_label(local)},"
                    f"error,error,{str(exc).replace(',', ';')}",
                    flush=True,
                )


def run_case(cl: OpenCL, case: ConvCase, iterations: int, variants: list[str], print_kernel_info: bool) -> None:
    _, weights, _, input_buf, weight_buf, bias_buf = upload_case(cl, case)
    output_size = case.output_channels * case.output_height * case.output_width
    output = np.empty((output_size,), dtype=np.float32)
    reference = None
    scalar_kernel = cl.build_kernel(SCALAR_KERNEL, "conv_scalar")
    if print_kernel_info:
        cl.print_kernel_capabilities(case, "scalar", scalar_kernel)

    print("case,variant,local_size,exec_median_ms,exec_min_ms,max_abs_diff", flush=True)

    if "scalar" in variants or "all" in variants:
        out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
        cl.set_args(scalar_kernel, common_args(case, input_buf, weight_buf, bias_buf, out_buf))
        local = (8, 4, 1)
        global_size = (round_up(case.output_width, local[0]), round_up(case.output_height, local[1]), case.output_channels)
        times = cl.run_kernel(scalar_kernel, global_size, local, iterations)
        cl.read_buffer(out_buf, output)
        reference = output.copy()
        print(
            f"{case.name},scalar,{local_label(local)},"
            f"{statistics.median(times):.3f},{min(times):.3f},0.000000000",
            flush=True,
        )

    if reference is None:
        out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
        cl.set_args(scalar_kernel, common_args(case, input_buf, weight_buf, bias_buf, out_buf))
        local = (8, 4, 1)
        global_size = (round_up(case.output_width, local[0]), round_up(case.output_height, local[1]), case.output_channels)
        cl.run_kernel(scalar_kernel, global_size, local, 1)
        cl.read_buffer(out_buf, output)
        reference = output.copy()

    local_candidates: tuple[tuple[int, int, int] | None, ...] = (
        None,
        (4, 4, 1),
        (8, 2, 1),
        (8, 4, 1),
        (16, 2, 1),
        (16, 4, 1),
        (8, 8, 1),
    )

    for block in (2, 4, 8, 16):
        packed = pack_weights(case, weights, block)
        packed_buf = cl.create_buffer(CL_MEM_READ_ONLY, packed.nbytes)
        cl.write_buffer(packed_buf, packed)
        for spatial in (1, 2, 4):
            variant = f"oc{block}_sp{spatial}_packed"
            if variant not in variants and "all" not in variants:
                continue
            source = (
                PACKED_KERNEL_TEMPLATE.replace("__BLOCK__", str(block)).replace(
                    "__SPATIAL__", str(spatial)
                )
            )
            kernel = cl.build_kernel(source, "conv_packed")
            if print_kernel_info:
                cl.print_kernel_capabilities(case, variant, kernel)
            out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
            cl.set_args(kernel, common_args(case, input_buf, packed_buf, bias_buf, out_buf))
            for local in local_candidates:
                if local is None:
                    global_size = (
                        (case.output_width + spatial - 1) // spatial,
                        case.output_height,
                        (case.output_channels + block - 1) // block,
                    )
                else:
                    global_size = (
                        round_up((case.output_width + spatial - 1) // spatial, local[0]),
                        round_up(case.output_height, local[1]),
                        round_up((case.output_channels + block - 1) // block, local[2]),
                    )
                try:
                    times = cl.run_kernel(kernel, global_size, local, iterations)
                    cl.read_buffer(out_buf, output)
                    max_abs = float(np.max(np.abs(output - reference)))
                    print(
                        f"{case.name},{variant},{local_label(local)},"
                        f"{statistics.median(times):.3f},{min(times):.3f},{max_abs:.9f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    print(
                        f"{case.name},{variant},{local_label(local)},"
                        f"error,error,{str(exc).replace(',', ';')}",
                        flush=True,
                    )

    if "oc16_vec" in variants or "all" in variants:
        packed = pack_weights(case, weights, 16)
        packed_buf = cl.create_buffer(CL_MEM_READ_ONLY, packed.nbytes)
        cl.write_buffer(packed_buf, packed)
        try:
            kernel = cl.build_kernel(OC16_VECTOR_KERNEL, "conv_oc16_vec")
            if print_kernel_info:
                cl.print_kernel_capabilities(case, "oc16_vec", kernel)
        except RuntimeError as exc:
            print(
                f"{case.name},oc16_vec,build,build_error,build_error,"
                f"{str(exc).replace(',', ';')}",
                flush=True,
            )
        else:
            out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
            cl.set_args(kernel, common_args(case, input_buf, packed_buf, bias_buf, out_buf))
            for local in (None, (32, 1, 1), (4, 4, 1), (8, 4, 1), (16, 2, 1), (16, 4, 1)):
                if local is None:
                    global_size = (
                        case.output_width,
                        case.output_height,
                        (case.output_channels + 15) // 16,
                    )
                else:
                    global_size = (
                        round_up(case.output_width, local[0]),
                        round_up(case.output_height, local[1]),
                        (case.output_channels + 15) // 16,
                    )
                try:
                    times = cl.run_kernel(kernel, global_size, local, iterations)
                    cl.read_buffer(out_buf, output)
                    max_abs = float(np.max(np.abs(output - reference)))
                    print(
                        f"{case.name},oc16_vec,{local_label(local)},"
                        f"{statistics.median(times):.3f},{min(times):.3f},{max_abs:.9f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    print(
                        f"{case.name},oc16_vec,{local_label(local)},"
                        f"error,error,{str(exc).replace(',', ';')}",
                        flush=True,
                    )

    if "oc16_ci4_vec" in variants or "all" in variants:
        packed = pack_weights(case, weights, 16)
        packed_buf = cl.create_buffer(CL_MEM_READ_ONLY, packed.nbytes)
        cl.write_buffer(packed_buf, packed)
        try:
            kernel = cl.build_kernel(OC16_CI4_VECTOR_KERNEL, "conv_oc16_ci4_vec")
            if print_kernel_info:
                cl.print_kernel_capabilities(case, "oc16_ci4_vec", kernel)
        except RuntimeError as exc:
            print(
                f"{case.name},oc16_ci4_vec,build,build_error,build_error,"
                f"{str(exc).replace(',', ';')}",
                flush=True,
            )
        else:
            out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
            cl.set_args(kernel, common_args(case, input_buf, packed_buf, bias_buf, out_buf))
            for local in (None, (32, 1, 1), (4, 4, 1), (8, 4, 1), (16, 2, 1), (16, 4, 1)):
                if local is None:
                    global_size = (
                        case.output_width,
                        case.output_height,
                        (case.output_channels + 15) // 16,
                    )
                else:
                    global_size = (
                        round_up(case.output_width, local[0]),
                        round_up(case.output_height, local[1]),
                        (case.output_channels + 15) // 16,
                    )
                try:
                    times = cl.run_kernel(kernel, global_size, local, iterations)
                    cl.read_buffer(out_buf, output)
                    max_abs = float(np.max(np.abs(output - reference)))
                    print(
                        f"{case.name},oc16_ci4_vec,{local_label(local)},"
                        f"{statistics.median(times):.3f},{min(times):.3f},{max_abs:.9f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    print(
                        f"{case.name},oc16_ci4_vec,{local_label(local)},"
                        f"error,error,{str(exc).replace(',', ';')}",
                        flush=True,
                    )

    if "oc16_ci4_ptr" in variants or "all" in variants:
        packed = pack_weights(case, weights, 16)
        packed_buf = cl.create_buffer(CL_MEM_READ_ONLY, packed.nbytes)
        cl.write_buffer(packed_buf, packed)
        try:
            kernel = cl.build_kernel(OC16_CI4_PTR_KERNEL, "conv_oc16_ci4_ptr")
            if print_kernel_info:
                cl.print_kernel_capabilities(case, "oc16_ci4_ptr", kernel)
        except RuntimeError as exc:
            print(
                f"{case.name},oc16_ci4_ptr,build,build_error,build_error,"
                f"{str(exc).replace(',', ';')}",
                flush=True,
            )
        else:
            out_buf = cl.create_buffer(CL_MEM_WRITE_ONLY, output.nbytes)
            cl.set_args(kernel, common_args(case, input_buf, packed_buf, bias_buf, out_buf))
            for local in (None, (32, 1, 1), (4, 4, 1), (8, 4, 1), (16, 2, 1), (16, 4, 1)):
                if local is None:
                    global_size = (
                        case.output_width,
                        case.output_height,
                        (case.output_channels + 15) // 16,
                    )
                else:
                    global_size = (
                        round_up(case.output_width, local[0]),
                        round_up(case.output_height, local[1]),
                        (case.output_channels + 15) // 16,
                    )
                try:
                    times = cl.run_kernel(kernel, global_size, local, iterations)
                    cl.read_buffer(out_buf, output)
                    max_abs = float(np.max(np.abs(output - reference)))
                    print(
                        f"{case.name},oc16_ci4_ptr,{local_label(local)},"
                        f"{statistics.median(times):.3f},{min(times):.3f},{max_abs:.9f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    print(
                        f"{case.name},oc16_ci4_ptr,{local_label(local)},"
                        f"error,error,{str(exc).replace(',', ';')}",
                        flush=True,
                    )


def bench_row_to_json(row: BenchRow) -> dict[str, object]:
    return {
        "case": row.case,
        "variant": row.variant,
        "local_size": row.local_size,
        "exec_median_ms": row.median_ms,
        "exec_min_ms": row.min_ms,
        "max_abs_diff": row.max_abs_diff,
    }


def write_autotune_summary_json(
    path: str,
    cl: OpenCL,
    recorder: CsvRecorder,
    cases: list[ConvCase],
    iterations: int,
    candidate_filter: list[str],
    local_filter: list[tuple[int, int, int] | None],
) -> None:
    best_rows = recorder.best_row_by_case()
    cases_payload = []
    for case in cases:
        rows = [row for row in recorder.rows if row.case == case.name]
        best = best_rows.get(case.name)
        cases_payload.append(
            {
                "case": asdict(case),
                "rows": [bench_row_to_json(row) for row in rows],
                "best": None if best is None else bench_row_to_json(best),
            }
        )

    payload = {
        "schema": "ov_gfx_opencl_conv_microbench_profile.v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "family": "oc16_ci4_ptr",
        "measurement": {
            "iterations": iterations,
            "summary_max_abs": recorder.summary_max_abs,
            "timing_kind": "event_profile" if cl.use_event_profiling else "host_wall_time",
        },
        "filters": {
            "candidates": candidate_filter,
            "local_sizes": [local_label(local) for local in local_filter],
        },
        "device": cl.profile_device_capabilities(),
        "cases": cases_payload,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASES), required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant to run: scalar, all, oc2_sp1_packed, oc4_sp2_packed, etc.",
    )
    parser.add_argument(
        "--kernel-info",
        action="store_true",
        help="Print clGetKernelWorkGroupInfo rows for each built kernel.",
    )
    parser.add_argument(
        "--autotune-oc16-ci4-ptr",
        action="store_true",
        help="Run a bounded source/workgroup autotune sweep around the oc16_ci4_ptr family.",
    )
    parser.add_argument(
        "--autotune-ci4-ptr-blocks",
        action="store_true",
        help="Run a bounded OC-block-size sweep around generated ci4_ptr kernels.",
    )
    parser.add_argument(
        "--autotune-candidate",
        action="append",
        default=[],
        help="Limit --autotune-oc16-ci4-ptr to named source candidates.",
    )
    parser.add_argument(
        "--autotune-local",
        action="append",
        type=parse_local_size,
        default=[],
        help="Limit --autotune-oc16-ci4-ptr to local sizes like auto or 16x1x2.",
    )
    parser.add_argument(
        "--allow-wall-time",
        action="store_true",
        help="Allow host wall-time timing if the target rejects CL_QUEUE_PROFILING_ENABLE.",
    )
    parser.add_argument(
        "--capabilities-only",
        action="store_true",
        help="Print OpenCL device capabilities and exit before building kernels.",
    )
    parser.add_argument(
        "--int8-dot-smoke",
        action="store_true",
        help="Run a GPU-only scalar-vs-packed int8 dot-product micro-smoke.",
    )
    parser.add_argument(
        "--int8-dot-count-words",
        type=int,
        default=262144,
        help="Number of packed int32 words for --int8-dot-smoke.",
    )
    parser.add_argument(
        "--summary-best",
        action="store_true",
        help="For --autotune-oc16-ci4-ptr, print the best valid row per case.",
    )
    parser.add_argument(
        "--summary-max-abs",
        type=float,
        default=1.0e-6,
        help="Max diff allowed in --summary-best rows.",
    )
    parser.add_argument(
        "--summary-json",
        help="Write a shape/caps/driver-keyed JSON profile artifact for --autotune-oc16-ci4-ptr.",
    )
    args = parser.parse_args()

    variants = args.variant or ["all"]
    if args.summary_json and not args.autotune_oc16_ci4_ptr:
        raise ValueError("--summary-json is only valid with --autotune-oc16-ci4-ptr")
    cl = OpenCL(allow_wall_time=args.allow_wall_time)
    cl.print_device_capabilities()
    if args.capabilities_only:
        return 0
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive unless --capabilities-only is used")
    if args.int8_dot_smoke:
        run_int8_dot_smoke(cl, args.int8_dot_count_words, args.iterations, args.kernel_info)
        return 0
    autotune_recorder = None
    if args.autotune_oc16_ci4_ptr:
        autotune_recorder = CsvRecorder(args.summary_best, args.summary_max_abs)
        autotune_recorder.header()
    for case_name in args.case:
        if args.autotune_ci4_ptr_blocks:
            run_ci4_ptr_block_autotune(cl, CASES[case_name], args.iterations, args.kernel_info)
        elif args.autotune_oc16_ci4_ptr:
            run_oc16_ci4_ptr_autotune(
                cl,
                CASES[case_name],
                args.iterations,
                args.kernel_info,
                set(args.autotune_candidate) if args.autotune_candidate else None,
                set(args.autotune_local) if args.autotune_local else None,
                autotune_recorder,
            )
        else:
            run_case(cl, CASES[case_name], args.iterations, variants, args.kernel_info)
    if autotune_recorder is not None:
        autotune_recorder.print_summary()
        if args.summary_json:
            write_autotune_summary_json(
                args.summary_json,
                cl,
                autotune_recorder,
                [CASES[name] for name in args.case],
                args.iterations,
                args.autotune_candidate,
                args.autotune_local,
            )
            print(f"summary_json,{args.summary_json}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
