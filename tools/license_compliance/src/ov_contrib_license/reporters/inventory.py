# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from ov_contrib_license.model import Inventory


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_lines(value: object, indentation: int = 0) -> list[str]:
    prefix = " " * indentation
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)) and child:
                lines.append(f"{prefix}{key}:")
                lines.extend(_yaml_lines(child, indentation + 2))
            elif isinstance(child, dict):
                lines.append(f"{prefix}{key}: {{}}")
            elif isinstance(child, list):
                lines.append(f"{prefix}{key}: []")
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(child)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]
        lines = []
        for child in value:
            if isinstance(child, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_lines(child, indentation + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(child)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]


def render_inventory(inventory: Inventory, output_format: str = "json") -> str:
    data = inventory.to_dict()
    if output_format == "json":
        return json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if output_format == "yaml":
        return "\n".join(_yaml_lines(data)) + "\n"
    raise ValueError(f"Unsupported inventory format: {output_format}")


def write_inventory(
    inventory: Inventory, output: Path | None, output_format: str = "json"
) -> None:
    rendered = render_inventory(inventory, output_format)
    if output is None:
        print(rendered, end="")
    else:
        output.write_text(rendered, encoding="utf-8")
