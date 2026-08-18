# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

MODULE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(MODULE_ROOT / "benchmarks"))
sys.path.insert(0, str(MODULE_ROOT / "genai_opt"))
