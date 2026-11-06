#!/usr/bin/env python3

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Export a trained PointPillars checkpoint to an OpenVINO IR.

The default export contains the neural network and uses ``pp_runtime`` for
voxelization and postprocessing. ``--embed-custom-ops`` includes those stages
in the graph and exposes ``points`` and ``num_points`` as model inputs.

The exporter also writes generated C++ headers, OpenCL sources, GPU kernel
configuration, and benchmark configuration.

The network uses half precision. Compile with ``EXECUTION_MODE_HINT=ACCURACY``
to preserve the precision written in the IR:

    core.compile_model(model, "GPU", {"EXECUTION_MODE_HINT": "ACCURACY"})

Each export variant is written to its own output directory with its model,
weights, and GPU configuration file.

Usage:

    python3 ov_export.py
    python3 ov_export.py --embed-custom-ops
    python3 ov_export.py --ckpt pretrained/epoch_160.pth --output-dir pretrained/ov
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as opset
from openvino.utils.node_factory import NodeFactory

# Use half precision for the network and single precision for other tensors.
NETWORK_TYPE = ov.Type.f16
NETWORK_DTYPE = np.float16

# Feature channels emitted for each point in a pillar.
PILLAR_FEATURES = ("x", "y", "z", "intensity",
                   "x_to_mean", "y_to_mean", "z_to_mean",
                   "x_to_centre", "y_to_centre", "z_to_centre")

# Feature channels consumed by the pillar encoder.
ENCODER_INPUTS = ("x_to_centre", "y_to_centre", "z", "intensity",
                  "x_to_mean", "y_to_mean", "z_to_mean",
                  "x_to_centre", "y_to_centre")

# Class and channel orders used by the exported decoder.
CHECKPOINT_CLASSES = ("pedestrian", "cyclist", "car")
DECODER_CLASSES = ("car", "pedestrian", "cyclist")
CHECKPOINT_BOX_VALUES = ("x", "y", "z", "width", "length", "height", "yaw")
DECODER_BOX_VALUES = ("x", "y", "z", "length", "width", "height", "yaw")
DIRECTION_VALUES = ("forward", "backward")

# Detection record layout used by the kernels and runtime. The class id occupies
# a float slot as raw integer bits.
DETECTION_RECORD = DECODER_BOX_VALUES + ("class_id", "score")

# Values stored for one anchor and one pillar coordinate record.
ANCHOR_VALUES = ("dx", "dy", "dz", "rotation")
COORD_RECORD = ("batch", "z", "y", "x")

# Number of anchors stored in one survivor-bitmap word.
SELECT_WORD_BITS = 32

# Epsilon used when folding batch normalization into convolution weights.
BATCH_NORM_EPS = 1e-3

# Configuration file containing scene, checkpoint, and device settings.
CONFIG_PATH = Path(__file__).resolve().parent / "ov_config.toml"


def _tuples(value):
    """Convert nested TOML lists to tuples."""
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def _load_config():
    """Load the exporter configuration."""
    if not CONFIG_PATH.is_file():
        raise SystemExit(f"{CONFIG_PATH} not found; it holds the scene geometry, the frame "
                         f"limits and the decoder constants")
    with CONFIG_PATH.open("rb") as handle:
        return tomllib.load(handle)


_CONFIG = _load_config()


def _setting(section, key):
    """Read one setting and convert nested lists to tuples."""
    try:
        return _tuples(_CONFIG[section][key])
    except KeyError:
        raise SystemExit(f"{CONFIG_PATH} is missing [{section}] {key}") from None


# Number of rotations assigned to each decoder class.
ROTATIONS_PER_CLASS = len(_setting("decoder", "rotations"))


@dataclass(frozen=True)
class PillarConfig:
    """Scene geometry and tensor capacities used by the export."""

    voxel_size: tuple = _setting("pillar", "voxel_size")
    point_cloud_range: tuple = _setting("pillar", "point_cloud_range")
    num_classes: int = _setting("pillar", "num_classes")
    backbone_stride: int = _setting("pillar", "backbone_stride")
    max_points: int = _setting("pillar", "max_points")
    max_pillars: int = _setting("pillar", "max_pillars")
    max_points_per_pillar: int = _setting("pillar", "max_points_per_pillar")
    point_features: int = _setting("pillar", "point_features")
    pfn_channels: int = _setting("pillar", "pfn_channels")

    def __post_init__(self):
        # The pillar representation requires a single voxel along z.
        if self.grid_z != 1:
            raise SystemExit(
                f"{CONFIG_PATH}: [pillar] voxel_size {list(self.voxel_size)} and point_cloud_range "
                f"{list(self.point_cloud_range)} give a grid {self.grid_z} voxels deep in z. "
                "PointPillars requires voxel_size.z to equal the z range.")

    @property
    def grid_x(self):
        """Number of grid cells along x."""
        min_x, _, _, max_x, _, _ = self.point_cloud_range
        voxel_x, _, _ = self.voxel_size
        return int((max_x - min_x) / voxel_x)

    @property
    def grid_y(self):
        """Number of grid cells along y."""
        _, min_y, _, _, max_y, _ = self.point_cloud_range
        _, voxel_y, _ = self.voxel_size
        return int((max_y - min_y) / voxel_y)

    @property
    def grid_z(self):
        """Number of grid cells along z."""
        _, _, min_z, _, _, max_z = self.point_cloud_range
        _, _, voxel_z = self.voxel_size
        return int((max_z - min_z) / voxel_z)

    @property
    def cells(self):
        """Number of cells in the 2D pseudo-image."""
        return self.grid_x * self.grid_y

    @property
    def feature_x(self):
        """Detection-head feature-map width."""
        return self.grid_x // self.backbone_stride

    @property
    def feature_y(self):
        """Detection-head feature-map height."""
        return self.grid_y // self.backbone_stride

    @property
    def anchors_per_cell(self):
        """Number of anchors at one feature-map cell."""
        return self.num_classes * ROTATIONS_PER_CLASS

    @property
    def anchor_count(self):
        """Total number of detection anchors."""
        return self.feature_x * self.feature_y * self.anchors_per_cell

    @property
    def select_words(self):
        """Number of words in the survivor bitmap."""
        return self.anchor_count // SELECT_WORD_BITS


@dataclass(frozen=True)
class DecoderConfig:
    """Decoder geometry, thresholds, and output capacities."""

    anchor_sizes: tuple = _setting("decoder", "anchor_sizes")
    anchor_bottom_heights: tuple = _setting("decoder", "anchor_bottom_heights")
    rotations: tuple = _setting("decoder", "rotations")
    score_threshold: float = _setting("decoder", "score_threshold")
    nms_threshold: float = _setting("decoder", "nms_threshold")
    dir_offset: float = _setting("decoder", "dir_offset")
    max_candidates: int = _setting("decoder", "max_candidates")
    max_detections: int = _setting("decoder", "max_detections")


@dataclass(frozen=True)
class TuningConfig:
    """OpenCL work-group sizes and packing settings."""

    scatter_wg: int = _setting("tuning", "scatter_wg")
    bev_row_wg: int = _setting("tuning", "bev_row_wg")
    bev_vec: int = _setting("tuning", "bev_vec")
    scores_wg: int = _setting("tuning", "scores_wg")
    candidates_wg: int = _setting("tuning", "candidates_wg")
    nms_block: int = _setting("tuning", "nms_block")
    nms_groups: int = _setting("tuning", "nms_groups")
    nms_simd: int = _setting("tuning", "nms_simd")
    gather_wg: int = _setting("tuning", "gather_wg")

    @property
    def nms_tile_wg(self):
        """Work-group size for one NMS tile."""
        return self.nms_block * self.nms_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _constant(values, dtype):
    return opset.constant(np.ascontiguousarray(values, dtype=dtype))


def _weight(values):
    """A network constant, in the network's element type."""
    return _constant(values, NETWORK_DTYPE)


def _folded_batch_norm(weight, state, prefix, channel_axis):
    """Fold batch-normalization scale and bias into convolution parameters."""
    scale = state[f"{prefix}.weight"].numpy().astype(np.float32)
    shift = state[f"{prefix}.bias"].numpy().astype(np.float32)
    mean = state[f"{prefix}.running_mean"].numpy().astype(np.float32)
    variance = state[f"{prefix}.running_var"].numpy().astype(np.float32)

    scale = scale / np.sqrt(variance + BATCH_NORM_EPS)
    broadcast = [1] * weight.ndim
    broadcast[channel_axis] = -1
    return weight * scale.reshape(broadcast), shift - mean * scale


def _conv_bn_relu(x, state, conv_key, norm_key, stride):
    """Convolution, batch norm and ReLU, folded into one convolution."""
    weight = state[conv_key].numpy().astype(np.float32)      # out, in, height, width
    weight, bias = _folded_batch_norm(weight, state, norm_key, channel_axis=0)
    padding = [(weight.shape[2] - 1) // 2, (weight.shape[3] - 1) // 2]

    x = opset.convolution(x, _weight(weight), [stride, stride], padding, padding, [1, 1])
    x = opset.add(x, _weight(bias.reshape(1, -1, 1, 1)))
    return opset.relu(x)


def _deconv_bn_relu(x, state, conv_key, norm_key):
    """Apply a folded transposed convolution, batch norm, and ReLU."""
    weight = state[conv_key].numpy().astype(np.float32)      # in, out, height, width
    weight, bias = _folded_batch_norm(weight, state, norm_key, channel_axis=1)

    x = opset.convolution_backprop_data(x, _weight(weight),
                                        strides=[weight.shape[2], weight.shape[3]],
                                        pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1])
    x = opset.add(x, _weight(bias.reshape(1, -1, 1, 1)))
    return opset.relu(x)


def _child_count(state, prefix):
    """How many numbered children a module has in the checkpoint."""
    numbered = {int(key[len(prefix):].split(".")[0]) for key in state if key.startswith(prefix)}
    return max(numbered) + 1 if numbered else 0


def _conv_indices(state, prefix):
    """Return sequence positions containing four-dimensional convolution weights."""
    return [index for index in range(_child_count(state, prefix))
            if f"{prefix}{index}.weight" in state
            and state[f"{prefix}{index}.weight"].dim() == 4]


# ---------------------------------------------------------------------------
# Generated configuration
# ---------------------------------------------------------------------------
GENERATED_BY = "Generated by ov_export.py. Edit [decoder] in ov_config.toml, not this file."
GENERATED_BY_PILLAR = "Generated by ov_export.py. Edit [pillar] in ov_config.toml, not this file."


def _guarded(names):
    """Wrap OpenCL defines in override guards."""
    lines = []
    for name, value in names.items():
        lines += [f"#ifndef {name}", f"#define {name} {value}", "#endif"]
    return lines


def _header_guard(filename):
    return f"__{Path(filename).stem.upper()}_HPP__"


def write_pillar_config(plugin_dir, config=PillarConfig()):
    """Write generated pillar geometry and metadata layout files."""
    minimum = config.point_cloud_range[:3]
    maximum = config.point_cloud_range[3:]

    def real(value):
        return f"{float(value)!r}f"

    kernel_values = {
        "MAX_POINTS": config.max_points,
        "MAX_VOXELS": config.max_pillars,
        "POINTS_PER_VOXEL": config.max_points_per_pillar,
        "POINT_FEATURES": config.point_features,
        "OUT_FEATURES": len(PILLAR_FEATURES),
        "GRID_X": config.grid_x,
        "GRID_Y": config.grid_y,
        "GRID_Z": config.grid_z,
        # BEV dimensions use the scatter kernel's names.
        "BEV_H": config.grid_y,
        "BEV_W": config.grid_x,
        "BEV_C": config.pfn_channels,
        "BEV_MAX_VOXELS": config.max_pillars,
    }

    kernel_values.update({f"MIN_{axis}": real(value) for axis, value in zip("XYZ", minimum)})
    kernel_values.update({f"MAX_{axis}": real(value) for axis, value in zip("XYZ", maximum)})
    kernel_values.update({f"VOXEL_{axis}": real(value)
                          for axis, value in zip("XYZ", config.voxel_size)})

    kernel = [f"// {GENERATED_BY_PILLAR}", ""] + _guarded(kernel_values)
    kernel += ["", "// Derived from the above; not overridable."]
    kernel += [f"#define {name} {value}" for name, value in _meta_layout(config).items()]

    header_name = "pp_pillar_config_generated.hpp"
    header_guard = _header_guard(header_name)

    # Generate the header content.
    header = [f"// {GENERATED_BY_PILLAR}", f"#ifndef {header_guard}",
              f"#define {header_guard}", "", "#include <cstdint>", "",
              "namespace PpExtension {", ""]
    header += [f"constexpr int64_t {name} = {value};" for name, value in (
        ("PP_MAX_POINTS", config.max_points),
        ("PP_MAX_VOXELS", config.max_pillars),
        ("PP_MAX_POINTS_PER_VOXEL", config.max_points_per_pillar),
        ("PP_GRID_X", config.grid_x),
        ("PP_GRID_Y", config.grid_y),
        ("PP_GRID_Z", config.grid_z),
        ("PP_NUM_POINT_FEATURES", config.point_features),
        ("PP_NUM_OUTPUT_FEATURES", len(PILLAR_FEATURES)),
        ("PP_PFN_CHANNELS", config.pfn_channels),
    )]
    header += [""]
    header += [f"constexpr float {name} = {real(value)};" for name, value in
               [(f"PP_RANGE_MIN_{axis}", value) for axis, value in zip("XYZ", minimum)] +
               [(f"PP_RANGE_MAX_{axis}", value) for axis, value in zip("XYZ", maximum)] +
               [(f"PP_VOXEL_SIZE_{axis}", value) for axis, value in zip("XYZ", config.voxel_size)]]
    header += ["", "// Derived from the above."]
    header += [f"constexpr int64_t PP_{name} = {value};"
               for name, value in _meta_layout(config).items()]
    header += ["", "}  // namespace PpExtension", "", f"#endif  // {header_guard}", ""]

    return _write(plugin_dir, {"pp_pillar_config_generated.cl": kernel, header_name: header})


def _meta_layout(config):
    """Return offsets and sizes for the voxelization metadata buffer."""
    counter, coord = 0, 1
    count = coord + config.max_pillars * len(COORD_RECORD)
    raw = count + config.max_pillars
    g2p = raw + config.max_pillars * config.max_points_per_pillar
    layout = {
        "GRID_SIZE": config.cells,
        "META_COUNTER": counter,
        "META_COORD": coord,
        "META_COUNT": count,
        "META_RAW": raw,
        "META_RAW_STRIDE": config.max_points_per_pillar,
        "META_G2P": g2p,
        "META_SIZE": g2p + config.cells,
        "COORD_VALUES": len(COORD_RECORD),
    }
    layout.update({f"COORD_{name.upper()}": index for index, name in enumerate(COORD_RECORD)})
    return layout


def _write(plugin_dir, files):
    directories = {".cl": plugin_dir / "pp_gpu" / "src" / "kernel_selector" / "cl_kernels",
                   ".hpp": plugin_dir / "pp_cpu" / "include",
                   ".xml": plugin_dir / "pp_gpu"}
    written = []
    for name, lines in files.items():
        path = directories[Path(name).suffix] / name
        # Ensure every generated file ends with one newline.
        path.write_text("\n".join(lines).rstrip("\n") + "\n")
        written.append(path)
    return written


def _decoder_constants(config, decoder, tuning):
    """Return decoder constants in the order consumed by the kernels."""
    order = [CHECKPOINT_CLASSES.index(name) for name in DECODER_CLASSES]

    anchors = []
    for index in order:
        width, length, height = decoder.anchor_sizes[index]
        for rotation in decoder.rotations:
            anchors.append((length, width, height, rotation))

    integers = {
        "PP_FEATURE_X": config.feature_x,
        "PP_FEATURE_Y": config.feature_y,
        "PP_NUM_CLASSES": config.num_classes,
        "PP_NUM_ROTATIONS": ROTATIONS_PER_CLASS,
        "PP_NUM_ANCHORS": config.anchors_per_cell,
        "PP_MAX_CANDIDATES": decoder.max_candidates,
        "PP_MAX_DETECTIONS": decoder.max_detections,
        "PP_NUM_BOX_VALUES": len(DECODER_BOX_VALUES),
        "PP_NUM_DIR_VALUES": len(DIRECTION_VALUES),
        "PP_ANCHOR_VALUES": len(ANCHOR_VALUES),
        "PP_DET_CHANNELS": len(DETECTION_RECORD),
        "PP_CLASS_INDEX": DETECTION_RECORD.index("class_id"),
        "PP_SCORE_INDEX": DETECTION_RECORD.index("score"),
        "PP_SELECT_BITS": SELECT_WORD_BITS,
        "PP_NMS_BLOCK": tuning.nms_block,
        # Derived.
        "PP_ANCHOR_COUNT": config.anchor_count,
        "PP_SELECT_WORDS": config.select_words,
        "PP_NMS_COL_BLOCKS": decoder.max_candidates // tuning.nms_block,
        "PP_NMS_TILE_WG": tuning.nms_tile_wg,
    }
    reals = {
        "PP_MIN_X": config.point_cloud_range[0],
        "PP_MAX_X": config.point_cloud_range[3],
        "PP_MIN_Y": config.point_cloud_range[1],
        "PP_MAX_Y": config.point_cloud_range[4],
        "PP_SCORE_THRESH": decoder.score_threshold,
        "PP_NMS_THRESH": decoder.nms_threshold,
        "PP_DIR_OFFSET": decoder.dir_offset,
    }
    bottom_heights = [decoder.anchor_bottom_heights[index] for index in order]
    return integers, reals, anchors, bottom_heights


def write_decoder_config(plugin_dir, config=PillarConfig(), decoder=DecoderConfig(),
                         tuning=TuningConfig()):
    """Write generated decoder constants for C++ and OpenCL."""
    integers, reals, anchors, bottom_heights = _decoder_constants(config, decoder, tuning)

    def real(value):
        return f"{value!r}f"

    rows = ",\n".join("    " + ", ".join(real(number) for number in anchor) for anchor in anchors)
    heights = ", ".join(real(number) for number in bottom_heights)

    kernel = [f"// {GENERATED_BY}", ""]
    kernel += [f"#define {name} {value}" for name, value in integers.items()]
    kernel += [f"#define {name} {real(value)}" for name, value in reals.items()]
    kernel += ["",
               f"// {{{', '.join(ANCHOR_VALUES)}}} per anchor.",
               "__constant float PP_ANCHORS[PP_NUM_ANCHORS * PP_ANCHOR_VALUES] = {", rows, "};",
               f"__constant float PP_ANCHOR_BOTTOM_HEIGHTS[PP_NUM_CLASSES] = {{{heights}}};", ""]

    header_name = "pp_decoder_config_generated.hpp"
    header_guard = _header_guard(header_name)
    header = [f"// {GENERATED_BY}", f"#ifndef {header_guard}",
              f"#define {header_guard}", "", "#include <cstdint>", "",
              "namespace PpExtension {", ""]
    header += [f"constexpr int64_t {name} = {value};" for name, value in integers.items()]
    header += [f"constexpr float {name} = {real(value)};" for name, value in reals.items()]
    header += ["",
               f"// {{{', '.join(ANCHOR_VALUES)}}} per anchor.",
               "constexpr float PP_ANCHORS[PP_NUM_ANCHORS * PP_ANCHOR_VALUES] = {", rows, "};",
               f"constexpr float PP_ANCHOR_BOTTOM_HEIGHTS[PP_NUM_CLASSES] = {{{heights}}};",
               "", "}  // namespace PpExtension", "", f"#endif  // {header_guard}", ""]

    return _write(plugin_dir, {"pp_decoder_config_generated.cl": kernel, header_name: header})


GENERATED_BY_TUNING = "Generated by ov_export.py. Edit [tuning] in ov_config.toml, not this file."


def _tuning_values(tuning):
    """``TuningConfig`` under the names the kernels use."""
    return {
        "SCATTER_WG": tuning.scatter_wg,
        "BEV_ROW_WG": tuning.bev_row_wg,
        "BEV_VEC": tuning.bev_vec,
        "SCORES_WG": tuning.scores_wg,
        "CAND_WG": tuning.candidates_wg,
        "NMS_GROUPS": tuning.nms_groups,
        "NMS_SIMD": tuning.nms_simd,
        "GATHER_WG": tuning.gather_wg,
    }


def write_tuning_config(plugin_dir, tuning=TuningConfig()):
    """Write generated OpenCL tuning constants and C++ launch constants."""
    values = _tuning_values(tuning)

    kernel = [f"// {GENERATED_BY_TUNING}", ""] + _guarded(values)

    header_name = "pp_tuning_config_generated.hpp"
    header_guard = _header_guard(header_name)
    header = [f"// {GENERATED_BY_TUNING}", f"#ifndef {header_guard}",
              f"#define {header_guard}", "", "#include <cstdint>", "",
              "namespace PpExtension {", ""]
    header += [f"constexpr size_t PP_{name} = {value};" for name, value in values.items()]
    header += ["", "}  // namespace PpExtension", "", f"#endif  // {header_guard}", ""]

    return _write(plugin_dir, {"pp_tuning_config_generated.cl": kernel, header_name: header})


# ---------------------------------------------------------------------------
# SimpleGPU custom layer configuration
# ---------------------------------------------------------------------------
KERNEL_DIR = "src/kernel_selector/cl_kernels"
ROUNDING = "-cl-fp32-correctly-rounded-divide-sqrt"

# Work sizes given as "B" are the output batch, which the plugin substitutes.
BATCH = "B"


def _custom_layers(config, tuning):
    """Describe the SimpleGPU custom ops and their launch sizes."""
    voxel = ["pp_pillar_config_generated.cl", "pp_tuning_config_generated.cl",
             "pp_voxelization_common.cl"]
    postproc = ["pp_decoder_config_generated.cl", "pp_tuning_config_generated.cl",
                "pp_postproc_common.cl"]

    def inputs(count):
        """``count`` inputs followed by the single output."""
        return [("input", port) for port in range(count)] + [("output", 0)]

    return [
        dict(name="PPVoxelizationScatter", entry="pp_voxelization_scatter",
             sources=voxel + ["pp_voxelization_scatter.cl"], tensors=inputs(2),
             rounding=True, global_size=tuning.scatter_wg, local_size=tuning.scatter_wg,
             comment="Voxelization stage 1: scatter points and compact pillars."),
        dict(name="PPVoxelizationFinalize", entry="pp_voxelization_finalize",
             sources=voxel + ["pp_voxelization_features.cl"], tensors=inputs(2),
             rounding=True, global_size=BATCH, local_size=None,
             comment="Voxelization stage 2: generate pillar features. One work-item per pillar."),
        dict(name="PPVoxelizationCoords", entry="pp_voxelization_coords",
             sources=voxel + ["pp_voxelization_coords.cl"], tensors=inputs(1),
             rounding=False, global_size=BATCH, local_size=None,
             comment="Voxelization stage 3: emit pillar coordinates. One work-item per pillar."),
        dict(name="PPVoxelizationParams", entry="pp_voxelization_params",
             sources=voxel + ["pp_voxelization_params.cl"], tensors=inputs(1),
             rounding=False, global_size=1, local_size=None,
             comment="Voxelization stage 4: emit the pillar count."),
        dict(name="PPVoxelizationCellPillar", entry="pp_voxelization_cells",
             sources=voxel + ["pp_voxelization_cells.cl"], tensors=inputs(1),
             rounding=False, global_size=BATCH, local_size=None,
             comment="Voxelization stage 5: emit the cell-to-pillar lookup. One work-item per cell."),
        dict(name="PPScatterBEV", entry="pp_scatter_bev",
             sources=["pp_pillar_config_generated.cl", "pp_tuning_config_generated.cl",
                      "pp_scatter_common.cl", "pp_scatter_bev.cl"],
             tensors=inputs(3), rounding=False,
             global_size=config.grid_y * tuning.bev_row_wg, local_size=tuning.bev_row_wg,
             comment="BEV stage: dense scatter. One work-group per BEV row."),
        dict(name="PPPostprocScores", entry="pp_postproc_scores",
             sources=postproc + ["pp_postproc_scores.cl"], tensors=inputs(1), rounding=True,
             global_size=config.anchor_count, local_size=tuning.scores_wg,
             comment="Postprocess stage 1: score filter. One work-item per anchor."),
        dict(name="PPPostprocCandidates", entry="pp_postproc_candidates",
             sources=postproc + ["pp_postproc_candidates.cl"], tensors=inputs(4), rounding=True,
             global_size=tuning.candidates_wg, local_size=tuning.candidates_wg,
             comment="Postprocess stage 2: compact, decode, and sort."),
        dict(name="PPPostprocNms", entry="pp_postproc_nms",
             sources=postproc + ["pp_postproc_nms.cl"], tensors=inputs(1), rounding=True,
             global_size=tuning.nms_groups * tuning.nms_tile_wg, local_size=tuning.nms_tile_wg,
             comment="Postprocess stage 3: rotated-BEV IoU bitmask."),
        dict(name="PPPostprocGather", entry="pp_postproc_gather",
             sources=postproc + ["pp_postproc_gather.cl"], tensors=inputs(2), rounding=True,
             global_size=tuning.gather_wg, local_size=tuning.gather_wg,
             comment="Postprocess stage 4: greedy suppression."),
    ]


def write_gpu_kernel_xml(plugin_dir, config=PillarConfig(), tuning=TuningConfig()):
    """Write the SimpleGPU custom-layer configuration."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f"<!-- {GENERATED_BY_TUNING} -->"]

    def size(value):
        return f"{value},1,1"

    for layer in _custom_layers(config, tuning):
        lines += [""]
        lines += [f'<!-- {layer["comment"]} -->']
        lines += [f'<CustomLayer name="{layer["name"]}" type="SimpleGPU" version="1">',
                  f'  <Kernel entry="{layer["entry"]}">']
        lines += [f'    <Source filename="{KERNEL_DIR}/{name}"/>' for name in layer["sources"]]
        lines += ["  </Kernel>", "  <Buffers>"]
        for index, (kind, port) in enumerate(layer["tensors"]):
            lines += [f'    <Tensor arg-index="{index}" type="{kind}" '
                      f'port-index="{port}" format="bfyx"/>']
        lines += ["  </Buffers>"]
        if layer["rounding"]:
            lines += [f'  <CompilerOptions options="{ROUNDING}"/>']
        sizes = f'global="{size(layer["global_size"])}"'
        if layer["local_size"] is not None:
            sizes += f' local="{size(layer["local_size"])}"'
        lines += [f"  <WorkSizes {sizes}/>", "</CustomLayer>"]

    return _write(plugin_dir, {"pp_custom_gpu_kernels.xml": lines})


# ---------------------------------------------------------------------------
# Graph stages
# ---------------------------------------------------------------------------
def _pillar_layer(factory, points, num_points):
    """Build the voxelization graph and return pillar features and cell ownership."""
    workspace = factory.create("PPVoxelizationScatter", [points, num_points])
    pillars = factory.create("PPVoxelizationFinalize", [workspace, points])
    cell_pillar = factory.create("PPVoxelizationCellPillar", [workspace])

    # Convert the custom-op output to the network element type.
    pillars = opset.convert(opset.convert(pillars, ov.Type.f32), NETWORK_TYPE)
    return pillars, cell_pillar


def _pillar_encoder(config, state, pillars, cell_pillar):
    """Encode each pillar and place the resulting features in a BEV image."""
    # Select the encoder's nine input channels from the ten generated channels.
    trained = state["pillar_encoder.conv.weight"].numpy().astype(np.float32)
    channels = trained.shape[0]
    trained = trained.reshape(channels, len(ENCODER_INPUTS))

    weight = np.zeros((channels, len(PILLAR_FEATURES)), np.float32)
    for encoder_channel, name in enumerate(ENCODER_INPUTS):
        weight[:, PILLAR_FEATURES.index(name)] += trained[:, encoder_channel]
    weight, bias = _folded_batch_norm(weight, state, "pillar_encoder.bn", channel_axis=0)

    features = opset.matmul(pillars, _weight(weight), False, True)
    features = opset.add(features, _weight(bias.reshape(1, 1, -1)))
    features = opset.relu(features)
    pooling_features = opset.reduce_max(features, _constant([1], np.int64), keep_dims=False)

    return _pseudo_image(config, pooling_features, cell_pillar, channels)


def _pseudo_image(config, pooling_features, cell_pillar, channels):
    """Gather pillar features into a dense BEV image."""
    columns = opset.transpose(pooling_features, _constant([1, 0], np.int64))
    columns = opset.concat(
        [_constant(np.zeros((channels, 1), NETWORK_DTYPE), NETWORK_DTYPE), columns], axis=1)

    canvas = opset.gather(columns, cell_pillar, _constant(1, np.int64))
    return opset.reshape(
        canvas, _constant([1, channels, config.grid_y, config.grid_x], np.int64), False)


def _backbone(config, state, x):
    """Downsampling blocks; the output of each one is kept for the neck."""
    outs = []
    for block in range(_child_count(state, "backbone.multi_blocks.")):
        prefix = f"backbone.multi_blocks.{block}."
        for position in _conv_indices(state, prefix):
            # Only the first convolution of a block shrinks the map.
            x = _conv_bn_relu(x, state, f"{prefix}{position}.weight", f"{prefix}{position + 1}",
                              stride=config.backbone_stride if position == 0 else 1)
        outs.append(x)
    return outs


def _neck(state, xs):
    """Bring every backbone output back to one size and stack them."""
    ups = [_deconv_bn_relu(xs[block], state, f"neck.decoder_blocks.{block}.0.weight",
                           f"neck.decoder_blocks.{block}.1")
           for block in range(_child_count(state, "neck.decoder_blocks."))]
    return opset.concat(ups, axis=1)


def _head_branch(config, state, x, prefix, values, decoder_values):
    """Build one head branch and flatten it in decoder order."""
    stride = len(values)
    order = [CHECKPOINT_CLASSES.index(name) * ROTATIONS_PER_CLASS * stride + rotation * stride
             + values.index(wanted)
             for name in DECODER_CLASSES
             for rotation in range(ROTATIONS_PER_CLASS)
             for wanted in decoder_values]

    weight = state[f"{prefix}.weight"].numpy().astype(np.float32)[order]
    bias = state[f"{prefix}.bias"].numpy().astype(np.float32)[order]

    x = opset.convolution(x, _weight(weight), [1, 1], [0, 0], [0, 0], [1, 1])
    x = opset.add(x, _weight(bias.reshape(1, -1, 1, 1)))
    x = opset.transpose(x, _constant([0, 2, 3, 1], np.int64))
    x = opset.convert(x, ov.Type.f32)
    return opset.reshape(
        x, _constant([config.feature_y * config.feature_x * len(order)], np.int64), False)


def _head(config, state, x):
    """Class scores, box offsets and direction scores for every anchor."""
    classes = tuple(range(config.num_classes))
    bbox_cls_pred = _head_branch(config, state, x, "head.conv_cls", classes, classes)
    bbox_pred = _head_branch(config, state, x, "head.conv_reg",
                             CHECKPOINT_BOX_VALUES, DECODER_BOX_VALUES)
    bbox_dir_cls_pred = _head_branch(config, state, x, "head.conv_dir_cls",
                                     DIRECTION_VALUES, DIRECTION_VALUES)
    return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


def _predicted_bboxes(factory, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred):
    """Score filter, box decode and non-maximum suppression, all on the device."""
    keep = factory.create("PPPostprocScores", [bbox_cls_pred])
    candidates = factory.create("PPPostprocCandidates",
                                [keep, bbox_cls_pred, bbox_pred, bbox_dir_cls_pred])
    overlaps = factory.create("PPPostprocNms", [candidates])
    return factory.create("PPPostprocGather", [candidates, overlaps])


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_model(state, config=PillarConfig(), extension_path=None):
    """Build the network-only or custom-op-embedded OpenVINO model."""
    factory = None
    if extension_path is None:
        pillars = opset.parameter(
            [config.max_pillars, config.max_points_per_pillar, len(PILLAR_FEATURES)],
            NETWORK_TYPE, name="pillars")
        cell_pillar = opset.parameter([config.cells], ov.Type.i32, name="cell_pillar")
        pillars.output(0).get_tensor().set_names({"pillars"})
        cell_pillar.output(0).get_tensor().set_names({"cell_pillar"})
        parameters = [pillars, cell_pillar]
    else:
        factory = NodeFactory()
        factory.add_extension(str(extension_path))

        points = opset.parameter([config.max_points, config.point_features],
                                 ov.Type.f32, name="points")
        num_points = opset.parameter([1], ov.Type.i32, name="num_points")
        points.output(0).get_tensor().set_names({"points"})
        num_points.output(0).get_tensor().set_names({"num_points"})
        parameters = [points, num_points]

        pillars, cell_pillar = _pillar_layer(factory, points, num_points)

        # Reject an extension built for a different grid.
        built_cells = int(cell_pillar.get_output_shape(0)[0])
        if built_cells != config.cells:
            raise SystemExit(
                f"{extension_path} was built for a grid of {built_cells} cells, but this "
                f"configuration asks for {config.cells}. Update PillarConfig or rebuild "
                f"ov_plugins with a matching voxel size and point cloud range.")

    pillar_features = _pillar_encoder(config, state, pillars, cell_pillar)
    xs = _backbone(config, state, pillar_features)
    x = _neck(state, xs)
    heads = _head(config, state, x)

    if factory is None:
        results = []
        for name, head in zip(("cls", "box", "dir"), heads):
            result = opset.result(head, name=name)
            result.output(0).get_tensor().set_names({name})
            results.append(result)
    else:
        result = opset.result(_predicted_bboxes(factory, *heads), name="detections")
        result.output(0).get_tensor().set_names({"detections"})
        results = [result]
    return ov.Model(results, parameters, "pointpillars")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", type=Path, default=root / "pretrained" / "epoch_160.pth",
                        help="trained checkpoint to read the weights from")
    parser.add_argument("--embed-custom-ops", action="store_true",
                        help="build the pillar layer and the decoder into the graph "
                             "instead of leaving them to pp_runtime")
    parser.add_argument("--output-dir", type=Path,
                        help="where to write <checkpoint name>.xml, .bin and "
                             "bench_gpu_config.json; defaults to pretrained/ov, or "
                             "pretrained/ov_e2e with --embed-custom-ops")
    parser.add_argument("--extension", type=Path,
                        default=root / "ov_plugins" / "build" / "pp_extension.so",
                        help="library holding the custom operations")
    parser.add_argument("--plugin-dir", type=Path, default=root / "ov_plugins",
                        help="where to write the constants the kernels compile against")
    parser.add_argument("--config-only", action="store_true",
                        help="write those constants and stop; this is what ov_plugins/build.sh "
                             "runs, and it needs neither the checkpoint nor a built extension")
    args = parser.parse_args()

    config = PillarConfig()
    tuning = TuningConfig()
    generated = write_pillar_config(args.plugin_dir, config)
    generated += write_decoder_config(args.plugin_dir, config, DecoderConfig(), tuning)
    generated += write_tuning_config(args.plugin_dir, tuning)
    generated += write_gpu_kernel_xml(args.plugin_dir, config, tuning)

    for path in generated:
        print(f"[ov_export]   -> {path}")

    if args.config_only:
        return

    if args.output_dir is None:
        args.output_dir = root / "pretrained" / ("ov_e2e" if args.embed_custom_ops else "ov")

    extension = args.extension if args.embed_custom_ops else None
    if extension is not None and not extension.is_file():
        raise SystemExit(f"{extension} not found. --embed-custom-ops puts the custom operations "
                         f"in the graph, so the extension has to exist first:\n"
                         f"  bash ov_plugins/build.sh && python3 ov_export.py --embed-custom-ops")

    # Import torch only for checkpoint loading.
    import torch

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)

    model = build_model(state, config, extension)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = args.output_dir / f"{args.ckpt.stem}.xml"
    ov.save_model(model, xml_path, compress_to_fp16=False)
    bin_path = xml_path.with_suffix(".bin")

    # The embedded graph requires the GPU custom-layer configuration.
    gpu = {"EXECUTION_MODE_HINT": "ACCURACY"}
    if args.embed_custom_ops:
        gpu["CONFIG_FILE"] = str(root / "ov_plugins" / "pp_gpu" / "pp_custom_gpu_kernels.xml")
    config_path = args.output_dir / "bench_gpu_config.json"
    config_path.write_text(json.dumps({"GPU": gpu}) + "\n")

    print(f"[ov_export] {args.ckpt}")
    print(f"[ov_export]   -> {xml_path}")
    print(f"[ov_export]   -> {bin_path} ({bin_path.stat().st_size / 1024 ** 2:.1f} MiB)")
    print(f"[ov_export]   -> {config_path}")

    for label, ports in (("in ", model.inputs), ("out", model.outputs)):
        for port in ports:
            print(f"[ov_export]   {label} {port.any_name:<12}"
                  f" {port.element_type.get_type_name():<4} {list(port.shape)}")


if __name__ == "__main__":
    main()
