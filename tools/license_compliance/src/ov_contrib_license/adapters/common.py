# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    pass


def read_json(path: Path, provider: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProviderError(
            f"Unable to read {provider} result {path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ProviderError(f"Invalid {provider} JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise ProviderError(f"{provider} result must contain a JSON object")
    return data
