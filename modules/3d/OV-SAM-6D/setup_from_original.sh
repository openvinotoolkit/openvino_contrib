#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Reconstruct the full SAM-6D working tree for the OpenVINO port.
#
# This repository (OV-SAM-6D) ships ONLY:
#   - New files created for the OpenVINO port (kept under SAM-6D/)
#   - Patches for the original files we modified (SAM-6D/patches/*.patch)
#
# Running this script will:
#   1. Clone the original SAM-6D repository at the pinned commit.
#   2. Overlay the original source files into SAM-6D/ WITHOUT overwriting
#      any of the OpenVINO port files already present.
#   3. Apply the ISM and PEM patches on top of the original files.
#
# After this, SAM-6D/ contains a complete, runnable tree. See OV_README.md.

set -euo pipefail

# --- Configuration -----------------------------------------------------------
ORIG_REPO_URL="https://github.com/JiehongLin/SAM-6D.git"
ORIG_REPO_COMMIT="1c2543b"

# Directory containing this script (= OV-SAM-6D root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Code root that holds the OpenVINO port files and patches.
CODE_DIR="$SCRIPT_DIR/SAM-6D"
PATCH_DIR="$CODE_DIR/patches"
MARKER="$CODE_DIR/.ov_setup_done"

# --- Pre-flight checks -------------------------------------------------------
command -v git >/dev/null 2>&1 || { echo "[ERROR] git is not installed."; exit 1; }

if [ ! -d "$CODE_DIR" ]; then
    echo "[ERROR] Expected code directory not found: $CODE_DIR"
    exit 1
fi

if [ ! -f "$PATCH_DIR/ism.patch" ] || [ ! -f "$PATCH_DIR/pem.patch" ]; then
    echo "[ERROR] Patch files not found in $PATCH_DIR"
    exit 1
fi

if [ -f "$MARKER" ]; then
    echo "[INFO] Setup already completed (found $MARKER). Nothing to do."
    echo "[INFO] Delete this marker and the restored original files to re-run."
    exit 0
fi

# --- 1. Clone the original repository ----------------------------------------
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "[INFO] Cloning original SAM-6D ($ORIG_REPO_COMMIT) ..."
git -c core.autocrlf=false clone --quiet "$ORIG_REPO_URL" "$TMP_DIR/SAM-6D"
git -C "$TMP_DIR/SAM-6D" -c core.autocrlf=false checkout --quiet "$ORIG_REPO_COMMIT"

ORIG_CODE_DIR="$TMP_DIR/SAM-6D/SAM-6D"
if [ ! -d "$ORIG_CODE_DIR" ]; then
    echo "[ERROR] Unexpected original repo layout: $ORIG_CODE_DIR not found."
    exit 1
fi

# --- 2. Overlay original files (do NOT clobber OpenVINO port files) ----------
# coreutils >= 9.2 warns that `cp -n` behavior is non-portable and recommends
# `--update=none` (available since coreutils 9.1). Prefer it when supported,
# falling back to `-n` on older GNU/BSD cp.
if cp --update=none --version >/dev/null 2>&1; then
    NOCLOBBER="--update=none"
else
    NOCLOBBER="-n"
fi

echo "[INFO] Restoring original source files (without overwriting port files) ..."
cp -r $NOCLOBBER "$ORIG_CODE_DIR/." "$CODE_DIR/"

# Also restore the original repo-root README.md and pics/ (they live one level
# above the code dir in the upstream repo). These are not committed here.
ORIG_REPO_ROOT="$TMP_DIR/SAM-6D"

# Append original repo files to the attribution-prefixed README
if [ -f "$ORIG_REPO_ROOT/README.md" ]; then
    cat "$ORIG_REPO_ROOT/README.md" >> "$SCRIPT_DIR/README.md"
fi

# Copy original pics (no overwrite)
[ -d "$ORIG_REPO_ROOT/pics" ] && cp -r $NOCLOBBER "$ORIG_REPO_ROOT/pics" "$SCRIPT_DIR/"

# --- 3. Apply patches --------------------------------------------------------
apply_patch() {
    local patch_file="$1"
    local prepared_patch="$patch_file"
    local tmp_patch=""
    local patch_targets

    # License comments may be prepended to patch files for CI checks.
    # git apply expects the diff stream to start at the patch header, so
    # strip any leading comment preamble before applying.
    if head -n 1 "$patch_file" | grep -q '^# '; then
        tmp_patch="$(mktemp)"
        awk '
            BEGIN { in_diff = 0 }
            {
                if (!in_diff) {
                    if ($0 ~ /^--- / || $0 ~ /^diff --git /) {
                        in_diff = 1
                        print
                    }
                    next
                }
                print
            }
        ' "$patch_file" > "$tmp_patch"
        prepared_patch="$tmp_patch"
    fi

    # Normalize CRLF -> LF on files touched by this patch to avoid
    # "different line endings" hunk failures on environments with autocrlf.
    patch_targets="$(awk '
        /^\+\+\+ / {
            path = $2
            if (path == "/dev/null") next
            sub(/^b\//, "", path)
            print path
        }
    ' "$prepared_patch" | sort -u)"

    if [ -n "$patch_targets" ]; then
        while IFS= read -r rel_path; do
            [ -z "$rel_path" ] && continue
            if [ -f "$CODE_DIR/$rel_path" ]; then
                sed -i 's/\r$//' "$CODE_DIR/$rel_path"
            fi
        done <<< "$patch_targets"
    fi

    echo "[INFO] Applying $(basename "$patch_file") ..."
    if git -C "$CODE_DIR" apply -p1 "$prepared_patch" 2>/dev/null; then
        [ -n "$tmp_patch" ] && rm -f "$tmp_patch"
        return 0
    fi
    # Fallback to GNU patch if git apply is unavailable/unhappy.
    if command -v patch >/dev/null 2>&1; then
        patch -p1 -d "$CODE_DIR" < "$prepared_patch"
        [ -n "$tmp_patch" ] && rm -f "$tmp_patch"
        return 0
    fi
    [ -n "$tmp_patch" ] && rm -f "$tmp_patch"
    echo "[ERROR] Failed to apply $patch_file"
    exit 1
}

apply_patch "$PATCH_DIR/ism.patch"
apply_patch "$PATCH_DIR/pem.patch"

# --- Done --------------------------------------------------------------------
touch "$MARKER"
echo ""
echo "[INFO] SAM-6D tree reconstructed successfully at: $CODE_DIR"
echo "[INFO] Next steps: see $SCRIPT_DIR/OV_README.md"
