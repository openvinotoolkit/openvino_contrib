#!/usr/bin/env python3
#
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Generate third-party-programs.txt from third-party-programs.json.

third-party-programs.json is a thin manifest: it stores the document
preamble, the separator string and a list of third party programs, each
described only by its display name, SPDX identifier and a URL pointing at
the upstream license. The license texts themselves are NOT stored in the
JSON - this script fetches them when rendering the file.

For every license the text is resolved in this order:

  1. url points at GitHub  -> download the license verbatim from the
     repository (github.com/.../blob/... is rewritten to raw.*).
  2. otherwise (e.g. an HTML page) -> download the canonical license text
     from the SPDX license list, keyed by the SPDX identifier. SPDX "WITH"
     expressions (license WITH exception) are assembled from the license
     and exception texts.
  3. neither works (proprietary EULA, license not in SPDX, ...) -> fall
     back to a local file shipped next to this script, referenced by the
     entry's optional "file" field, under licenses/.
  4. none of the above -> raise an error and abort.

The rendered layout is:

    <preamble>

    -------------------------------------------------------------
    <marker><index> <name>

    <license text>

    -------------------------------------------------------------
    ...

Each entry header is "<marker><index> <name>" placed right after the
separator line, so a license can be located by its index (e.g. "№42").

Usage:
    python .github/scripts/generate_third_party_programs.py
    python .github/scripts/generate_third_party_programs.py --stdout
"""

import argparse
import json
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# This script lives in .github/scripts/third-party-programs/; the
# repository root is three levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
# The manifest sits next to this script; the generated inventory is
# written to the repository root.
JSON_PATH = SCRIPT_DIR / "third-party-programs.json"
TXT_PATH = ROOT / "third-party-programs.txt"
# Local fallback texts for licenses that cannot be fetched (proprietary
# EULAs, licenses missing from the SPDX list, ...).
LOCAL_LICENSES_DIR = SCRIPT_DIR / "licenses"

# Marker prefixed to the running index in every entry header (the numero
# sign). It is not used anywhere in the license texts, so "№42" uniquely
# locates the 42nd license.
INDEX_MARKER = "№"

# Canonical SPDX license texts (and exception texts).
SPDX_TEXT_BASE = "https://raw.githubusercontent.com/spdx/license-list-data/main/text"

USER_AGENT = "openvino_contrib-third-party-programs-generator"


def _http_get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8", "replace")
    if body.lstrip().startswith("404:"):
        raise RuntimeError(f"got a 404 body from {url}")
    return body


def _is_github(url):
    return "github.com" in url or "raw.githubusercontent.com" in url


def _github_raw(url):
    """Rewrite a github.com blob URL to its raw.githubusercontent.com form."""
    if "github.com" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/")
        url = url.replace("/blob/", "/", 1)
    return url


class _TextExtractor(HTMLParser):
    """Collect human-visible text from an HTML document."""

    _SKIP = {"script", "style", "head", "nav", "header", "footer"}

    def __init__(self):
        super().__init__()
        self._chunks = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self._chunks.append(data.strip())

    def text(self):
        return "\n".join(self._chunks)


def _fetch_spdx(spdx):
    """Fetch canonical SPDX text; supports 'LICENSE WITH EXCEPTION'."""
    if spdx.startswith("LicenseRef-"):
        raise RuntimeError(f"SPDX id '{spdx}' is a LicenseRef, not in the SPDX list")
    base, _, exception = spdx.partition(" WITH ")
    text = _http_get(f"{SPDX_TEXT_BASE}/{base.strip()}.txt")
    if exception.strip():
        exc_text = _http_get(f"{SPDX_TEXT_BASE}/exceptions/{exception.strip()}.txt")
        text = f"{text.rstrip()}\n\n{exc_text}"
    return text


def _read_local(entry):
    """Read a local fallback license file referenced by entry['file']."""
    name = entry.get("file")
    if not name:
        return None
    path = LOCAL_LICENSES_DIR / name
    if not path.is_file():
        raise RuntimeError(f"local license file not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_text(entry):
    """Resolve a license text following the url -> spdx -> file order."""
    name = entry["name"]
    url = entry.get("url", "")
    spdx = entry.get("spdx", "")

    # 1. GitHub-hosted license -> verbatim download.
    if url and _is_github(url):
        return _http_get(_github_raw(url)).rstrip("\n")

    # 2. Non-GitHub URL -> canonical SPDX text.
    if spdx:
        try:
            return _fetch_spdx(spdx).rstrip("\n")
        except Exception as spdx_error:
            # 3. Fall back to a local file if one is provided.
            local = _read_local(entry)
            if local is not None:
                return local.rstrip("\n")
            raise RuntimeError(
                f"cannot resolve license for '{name}': SPDX lookup failed "
                f"({spdx_error}) and no local 'file' is provided"
            )

    # 3. No usable url/spdx -> local file is the last resort.
    local = _read_local(entry)
    if local is not None:
        return local.rstrip("\n")

    # 4. Nothing worked.
    raise RuntimeError(f"cannot resolve license text for '{name}'")


def render(doc):
    """Render the manifest into the third-party-programs.txt text."""
    separator = doc["separator"]
    licenses = doc["licenses"]

    lines = []
    lines.extend(doc["preamble"].split("\n"))
    lines.append("")  # blank line between preamble and first separator

    last = len(licenses) - 1
    for i, entry in enumerate(licenses):
        index = i + 1  # 1-based running index used in the header
        text = resolve_text(entry)
        header = f"{INDEX_MARKER}{index} {entry['name']}"
        lines.append(separator)
        lines.append(header)
        lines.append("")  # blank line between header and license text
        lines.extend(text.split("\n"))
        if i != last:
            lines.append("")  # blank line separating from the next separator

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the generated content instead of writing the file",
    )
    args = parser.parse_args()

    with JSON_PATH.open(encoding="utf-8") as handle:
        doc = json.load(handle)

    print(f"Resolving {len(doc['licenses'])} license texts ...", file=sys.stderr)
    generated = render(doc)

    if args.stdout:
        sys.stdout.write(generated)
        return

    TXT_PATH.write_text(generated, encoding="utf-8")
    print(f"Wrote {TXT_PATH.name} ({len(doc['licenses'])} licenses).")


if __name__ == "__main__":
    main()
