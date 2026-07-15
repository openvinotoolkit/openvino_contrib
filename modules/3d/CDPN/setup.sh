#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Prepares the `cdpn_repo` for OpenVINO inference: clones the upstream repo
# (pinned to commit 625f9a8), applies cdpn_changes.patch, and copies the
# OpenVINO-specific files listed in copy_files_to_cdpn_repo.txt.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$HERE/cdpn_repo"
UPSTREAM_URL="https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi"
UPSTREAM_COMMIT="625f9a8"
PATCH="$HERE/cdpn_changes.patch"
COPY_LIST="$HERE/copy_files_to_cdpn_repo.txt"
REQUIREMENTS="$HERE/ov_requirements.txt"

cd "$HERE"

# 1. Clone only if missing.
if [ -d "$REPO_DIR/.git" ]; then
  echo "cdpn_repo already present, skipping clone."
else
  echo "Cloning $UPSTREAM_URL ..."
  git clone "$UPSTREAM_URL" "$REPO_DIR"
fi

# 2. Pin to the known-good commit.
git -C "$REPO_DIR" checkout "$UPSTREAM_COMMIT"

# 3. Apply the patch if it is not already applied.
if patch -p1 --forward --dry-run -d "$REPO_DIR" < "$PATCH" >/dev/null 2>&1; then
  echo "Applying patch ..."
  patch -p1 --forward -d "$REPO_DIR" < "$PATCH"
else
  echo "Patch already applied; skipping."
fi

# 4. Copy new files.
grep -v -E '^[[:space:]]*(#|$)' "$COPY_LIST" | while IFS= read -r f; do
  [ -n "$f" ] || continue
  mkdir -p "$REPO_DIR/$(dirname "$f")"
  cp "$HERE/$f" "$REPO_DIR/$f"
done

# 5. Install Python requirements.
python -m pip install --upgrade -r "$REQUIREMENTS"
