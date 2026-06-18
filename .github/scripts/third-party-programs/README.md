<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Third party programs inventory

This directory owns everything needed to (re)generate the repository-level
[`third-party-programs.txt`](../../../third-party-programs.txt) — the file that
lists every third party program bundled with or fetched by the modules in this
repository, together with its license text.

## Contents

| Object | Purpose |
| --- | --- |
| `generate_third_party_programs.py` | Generator: reads the manifest, fetches every license text and writes `third-party-programs.txt`. |
| `third-party-programs.json` | Manifest (single source of truth): the document preamble, the separator and the list of programs. |
| `licenses/` | Verbatim license texts that cannot be fetched automatically (e.g. proprietary EULAs). |
| `README.md` | This file. |

## How `third-party-programs.txt` is produced

The manifest deliberately stores **no license texts** — only, for each program,
its `name`, `spdx` identifier and a `url`. The generator resolves the actual
text for every entry in this order:

1. **`url` points at GitHub** → the license is downloaded verbatim from the
   repository (`github.com/.../blob/...` is rewritten to `raw.githubusercontent.com`).
2. **otherwise** (e.g. the `url` is an HTML page) → the canonical license text is
   downloaded from the [SPDX license list](https://github.com/spdx/license-list-data)
   using the `spdx` identifier. `WITH` expressions (license + exception) are
   assembled from both texts.
3. **neither works** (proprietary EULA, license missing from SPDX, …) → the text
   is read from a local file in `licenses/`, referenced by the entry's optional
   `file` field.
4. **nothing resolves** → the generator fails with an error.

Each entry is then numbered (`№1`, `№2`, …) and the entries are joined under the
shared preamble to form `third-party-programs.txt`.

## Regenerating the file

Requires Python 3 and network access (license texts are fetched on the fly).

```bash
# from anywhere in the repository
python .github/scripts/third-party-programs/generate_third_party_programs.py
```

This overwrites `third-party-programs.txt` at the repository root. To preview the
result without touching the file, append `--stdout`:

```bash
python .github/scripts/third-party-programs/generate_third_party_programs.py --stdout
```

## Adding or changing a program

Edit `third-party-programs.json` only, then regenerate:

1. Add an object to the `licenses` array with `name`, `spdx` and `url`.
   - Prefer a GitHub `url` that points straight at the `LICENSE` file.
   - For a license that cannot be fetched, drop its text into `licenses/<file>.txt`
     and add a `"file": "<file>.txt"` field to the entry.
2. Run the generator and commit both `third-party-programs.json` and the
   regenerated `third-party-programs.txt`.

Entry order in the JSON defines the order — and therefore the `№` numbering — in
the generated file.
