#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

base_ref=${1:-master}
formatter="$(command -v git-clang-format-18 || command -v git-clang-format || true)"
binary="$(command -v clang-format-18 || command -v clang-format || true)"
if [[ -z "$formatter" || -z "$binary" ]]; then
	echo "clang-format is not available" >&2
	exit 1
fi

"$formatter" --binary "$binary" "origin/${base_ref}" -- modules/nvidia_plugin
git diff --exit-code -- modules/nvidia_plugin
