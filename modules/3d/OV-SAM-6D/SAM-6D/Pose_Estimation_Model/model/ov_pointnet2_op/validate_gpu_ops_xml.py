#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validate pem_gpu_ops.xml has format='any' on multi-port inputs to avoid reorder naming collisions."""
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

XML_PATH = Path(__file__).parent / "pem_gpu_ops.xml"

# Ops that have multiple inputs sharing the same source tensor (need format="any")
MULTI_INPUT_OPS = {"GatherOperation", "GroupingOperation", "BallQuery", "FurthestPointSampling"}


def validate():
    # The file has multiple root elements (no single root), wrap in a synthetic root
    content = XML_PATH.read_text()
    root = ET.fromstring(f"<root>{content}</root>")
    errors = []
    for layer in root.iter("CustomLayer"):
        name = layer.get("name", "")
        if name not in MULTI_INPUT_OPS:
            continue
        for buf in layer.iter("Tensor"):
            arg_idx = buf.get("arg-index", "")
            port_idx = buf.get("port-index", "")
            fmt = buf.get("format", "")
            # Input tensors (port-index == arg-index for inputs) that lack format="any"
            if port_idx == arg_idx and fmt != "any":
                errors.append(f"{name}: input port {port_idx} has format='{fmt}' (expected 'any')")

    if errors:
        print("ERROR: pem_gpu_ops.xml validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print("pem_gpu_ops.xml: OK (all multi-port inputs have format='any')")


if __name__ == "__main__":
    validate()
