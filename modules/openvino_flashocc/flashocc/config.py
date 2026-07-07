# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Reimplementation of FlashOCC under Apache-2.0-compatible terms.

import runpy
from pathlib import Path


def load_config(path):
    cfg_path = Path(path)
    namespace = runpy.run_path(str(cfg_path))
    if "model" not in namespace:
        raise KeyError(f"No 'model' entry found in config: {cfg_path}")
    return namespace
