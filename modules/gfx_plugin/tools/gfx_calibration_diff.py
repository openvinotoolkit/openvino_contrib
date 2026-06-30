#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_probe_map(artifact: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    probes = artifact.get("probes", [])
    return {probe.get("name", f"probe_{idx}"): probe for idx, probe in enumerate(probes)}


def set_delta(before: List[str], after: List[str]) -> Dict[str, List[str]]:
    before_set = set(before)
    after_set = set(after)
    return {
        "added": sorted(after_set - before_set),
        "removed": sorted(before_set - after_set),
        "unchanged": sorted(before_set & after_set),
    }


def numeric_delta(old: float, new: float) -> Dict[str, Any]:
    abs_delta = new - old
    rel_delta = 0.0 if old == 0 else abs_delta / old
    return {
        "old": old,
        "new": new,
        "abs_delta": abs_delta,
        "rel_delta": rel_delta,
    }


def compare_probes(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    metric_names = [
        "arithmetic_intensity",
        "overhead_subtracted_ms",
        "adjusted_gbps",
        "adjusted_tflops",
        "gpu_gbps",
        "gpu_tflops",
        "first_to_steady_ratio",
        "wait_share_of_wall",
        "transfer_share_of_wall",
    ]
    deltas = {name: numeric_delta(float(before.get(name, 0.0)), float(after.get(name, 0.0))) for name in metric_names}
    return {
        "actual_backend_before": before.get("actual_backend", ""),
        "actual_backend_after": after.get("actual_backend", ""),
        "submit_count": numeric_delta(float(before.get("submit_count", 0)), float(after.get("submit_count", 0))),
        "barrier_count": numeric_delta(float(before.get("barrier_count", 0)), float(after.get("barrier_count", 0))),
        "metrics": deltas,
        "hints": set_delta(before.get("hints", []), after.get("hints", [])),
    }


def build_summary(diff: Dict[str, Any]) -> List[str]:
    summary: List[str] = []
    if not diff["device_key_match"]:
        summary.append("device_key_mismatch")
    if not diff["backend_match"]:
        summary.append("backend_mismatch")

    fixed_overhead_rel = diff["top_level"]["fixed_overhead_us"]["rel_delta"]
    if fixed_overhead_rel >= 0.10:
        summary.append("fixed_overhead_regressed")
    elif fixed_overhead_rel <= -0.10:
        summary.append("fixed_overhead_improved")

    compute_rel = diff["top_level"]["compute_estimate_tflops"]["rel_delta"]
    if compute_rel >= 0.10:
        summary.append("compute_estimate_improved")
    elif compute_rel <= -0.10:
        summary.append("compute_estimate_regressed")

    bandwidth_rel = diff["top_level"]["bandwidth_estimate_gbps"]["rel_delta"]
    if bandwidth_rel >= 0.10:
        summary.append("bandwidth_estimate_improved")
    elif bandwidth_rel <= -0.10:
        summary.append("bandwidth_estimate_regressed")

    probe_mb3 = diff["probes"].get("MB3")
    if probe_mb3:
        wait_rel = probe_mb3["metrics"]["wait_share_of_wall"]["rel_delta"]
        if wait_rel <= -0.10:
            summary.append("mb3_wait_share_improved")
        elif wait_rel >= 0.10:
            summary.append("mb3_wait_share_regressed")
    return summary


def compare_artifacts(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_probes = get_probe_map(before)
    after_probes = get_probe_map(after)
    probe_names = sorted(set(before_probes) | set(after_probes))

    result: Dict[str, Any] = {
        "before_device_key": before.get("device_key", ""),
        "after_device_key": after.get("device_key", ""),
        "device_key_match": before.get("device_key") == after.get("device_key"),
        "backend_match": before.get("backend") == after.get("backend"),
        "schema_match": (
            before.get("schema_version") == after.get("schema_version")
            and before.get("microbench_schema_version") == after.get("microbench_schema_version")
        ),
        "top_level": {
            "fixed_overhead_us": numeric_delta(float(before.get("fixed_overhead_us", 0.0)),
                                               float(after.get("fixed_overhead_us", 0.0))),
            "bandwidth_estimate_gbps": numeric_delta(float(before.get("bandwidth_estimate_gbps", 0.0)),
                                                     float(after.get("bandwidth_estimate_gbps", 0.0))),
            "compute_estimate_tflops": numeric_delta(float(before.get("compute_estimate_tflops", 0.0)),
                                                     float(after.get("compute_estimate_tflops", 0.0))),
            "gpu_bandwidth_estimate_gbps": numeric_delta(float(before.get("gpu_bandwidth_estimate_gbps", 0.0)),
                                                         float(after.get("gpu_bandwidth_estimate_gbps", 0.0))),
            "gpu_compute_estimate_tflops": numeric_delta(float(before.get("gpu_compute_estimate_tflops", 0.0)),
                                                         float(after.get("gpu_compute_estimate_tflops", 0.0))),
        },
        "triage_hints": set_delta(before.get("triage_hints", []), after.get("triage_hints", [])),
        "assumptions": set_delta(before.get("assumptions", []), after.get("assumptions", [])),
        "probes": {},
        "missing_probes": {
            "only_in_before": sorted(set(before_probes) - set(after_probes)),
            "only_in_after": sorted(set(after_probes) - set(before_probes)),
        },
    }

    for probe_name in probe_names:
        if probe_name not in before_probes or probe_name not in after_probes:
            continue
        result["probes"][probe_name] = compare_probes(before_probes[probe_name], after_probes[probe_name])

    result["summary_flags"] = build_summary(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two GFX microbench calibration artifacts.")
    parser.add_argument("--before", required=True, help="Path to the baseline calibration artifact JSON.")
    parser.add_argument("--after", required=True, help="Path to the new calibration artifact JSON.")
    parser.add_argument("--output", help="Optional path to write the diff JSON.")
    parser.add_argument("--fail-on-device-mismatch", action="store_true",
                        help="Return a non-zero exit code when device keys differ.")
    args = parser.parse_args()

    before = load_json(Path(args.before))
    after = load_json(Path(args.after))
    diff = compare_artifacts(before, after)
    payload = json.dumps(diff, indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")

    print(payload)
    if args.fail_on_device_mismatch and not diff["device_key_match"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
