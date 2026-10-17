<!--
Copyright (C) 2018-2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Contrib License Compliance

`ov-contrib-license` builds a deterministic third-party component inventory,
imports evidence from FOSS scanners, applies repository-owned policy, reconciles
third-party notices, emits SPDX, and compares built artifacts with source
expectations. It is independent of the OpenVINO build and runtime.

The checker is a policy automation tool, not a legal opinion. Unknown evidence
is preserved as `REVIEW` or `FAIL`; it is never silently treated as compatible.

## End-to-end quick start

Always install and run the tool in a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ./tools/license_compliance
.venv/bin/python -m unittest discover -s tools/license_compliance/tests -v
```

Run the same source-policy check used by CI:

```bash
.venv/bin/ov-contrib-license check . \
  --offline \
  --fail-on FAIL \
  --inventory-output inventory.json \
  --report license-report.md
```

For a PR-sized check, discovery expands changed files to their owning module or
repository scope:

```bash
.venv/bin/ov-contrib-license check . \
  --base-ref origin/master \
  --head-ref HEAD \
  --offline \
  --fail-on FAIL \
  --inventory-output inventory.json \
  --report license-report.md
```

Policy lives under `policy/`. The initial rollout blocks unsuppressed `FAIL`
findings while leaving insufficient evidence visible as `REVIEW`. Passing
`--fail-on REVIEW` enables the stricter gate.

## Evidence providers

Repository discovery is built in and offline. Previously generated ORT and
License Eye JSON can enrich the same canonical inventory:

```bash
.venv/bin/ov-contrib-license check . \
  --ort-result ort-result.json \
  --license-eye-result license-eye-result.json \
  --inventory-output inventory.json
```

Provider errors use a distinct infrastructure exit code and cannot become a
policy pass. ORT and License Eye are optional executables: the core never
downloads tools or contacts a SaaS backend. Syft can either run locally against
an explicit artifact or consume a saved result:

```bash
.venv/bin/ov-contrib-license artifact check build/install \
  --source-inventory inventory.json \
  --report artifact-report.md

.venv/bin/ov-contrib-license artifact check build/install \
  --source-inventory inventory.json \
  --syft-result syft-result.json
```

Artifact reconciliation reports undeclared packages, missing expected runtime
packages, and source/artifact license mismatches. Build-, test-, and dev-only
dependencies are not expected in runtime artifacts.

## Repository discovery

The inventory command is also available without policy evaluation:

```bash
.venv/bin/ov-contrib-license inventory . --output inventory.json
.venv/bin/ov-contrib-license inventory . --format yaml
```

Discovery covers:

- Python, Node.js, Go, Gradle, Cargo, Conan, Docker, and Git manifests;
- balanced, multiline CMake dependency commands without evaluating conditions;
- executable `git clone`, `curl`, `wget`, and remote package-install commands;
- Git submodules, including their index commit when available;
- vendored-source directories, nested license files, and foreign-copyright clusters;
- local, container, and repository-backed GitHub Actions.

Additional `--include` and `--exclude` globs may be repeated. Canonical output
uses repository-relative paths, stable ordering, and no timestamps.

## Policy, exceptions, and baseline

The policy directory contains:

```text
licenses.yml     maintainer-approved license classifications
rules.yml        relationship- and distribution-aware decisions
obligations.yml  attribution/source obligations
exceptions.yml   exact, approved, expiring exceptions
baseline.yml     exact historical finding fingerprints
toolchain.yml    FOSS provider/action policy and curated license registry
```

Policy evaluates the complete SPDX expression. `OR` alternatives are not
flattened into simultaneous obligations, while `AND` branches are all applied.
Exceptions must pin component version/revision, module, relationship,
distribution, approver, rationale, and expiration. A baseline suppresses only
the exact fingerprint; changing version, relationship, distribution, module, or
rule creates a new finding.

Use `explain` with an inventory component or finding ID:

```bash
.venv/bin/ov-contrib-license explain pkg:pypi/example@1.2.3 \
  --inventory inventory.json
```

## TPP and SPDX

`third-party-programs.txt` remains human-reviewed and authoritative. The tool
checks evidence-driven coverage and can generate a preview without overwriting
the repository file:

```bash
.venv/bin/ov-contrib-license tpp check . \
  --source-inventory inventory.json \
  --report tpp-report.md
.venv/bin/ov-contrib-license tpp generate . \
  --source-inventory inventory.json \
  --output /tmp/third-party-programs.preview.txt
```

Generate a deterministic SPDX 2.3 JSON document from the same inventory:

```bash
.venv/bin/ov-contrib-license sbom . \
  --source-inventory inventory.json \
  --format spdx-json \
  --output openvino-contrib.spdx.json
```

## Toolchain audit and CI

The self-audit checks immutable GitHub Action pins, curated FOSS licenses,
direct Python dependencies, and forbidden proprietary decision backends:

```bash
.venv/bin/ov-contrib-license toolchain check . --fail-on FAIL
```

`.github/workflows/license_compliance.yml` runs the offline suite and PR
fast-path, writes the Markdown reports to the GitHub step summary, and uploads
the canonical inventory, reports, and SPDX output. Artifact checking remains a
separate command because the compliance module does not build OpenVINO; build
jobs should pass an existing install tree or package to it.

## Exit codes

```text
0  no configured blocking findings
2  unsuppressed policy FAIL
3  REVIEW is configured as blocking
4  invalid arguments, policy, or schema
5  discovery/provider infrastructure failure
6  unexpected internal error
```

The direct Python dependencies are PyYAML 6.0.3 (MIT), used to load explicit
version-controlled policy, and Tomli 2.2.1 (MIT) on Python 3.10. Optional ORT,
License Eye, and Syft executables are not installed or downloaded by the package.
