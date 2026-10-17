#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PERF_STAT_RE = re.compile(r"^\s*([0-9,\.]+)\s+([A-Za-z0-9_\-\.]+)(?:\s+#\s+(.*))?$")
PERF_REPORT_RE = re.compile(r"^\s*([0-9]+\.[0-9]+)%\s+.*?\s+(\S+)$")


def parse_number(token: str) -> float:
    return float(token.replace(",", ""))


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def detect_kind(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        if '"traceEvents"' in text:
            return "trace_event"
        return "json"
    if "cache-misses" in text or "cycles" in text:
        return "perf_stat"
    if "Children" in text or "Self" in text or PERF_REPORT_RE.search(text):
        return "perf_report"
    return "text"


def summarize_trace_events(obj: Dict[str, Any]) -> Dict[str, Any]:
    events = obj.get("traceEvents", [])
    phase_totals: Dict[str, float] = defaultdict(float)
    event_counts: Dict[str, int] = defaultdict(int)
    hot_events: List[Tuple[float, str, str]] = []
    timestamps: List[float] = []

    for event in events:
        if not isinstance(event, dict):
            continue
        phase = str(event.get("cat", "unknown"))
        name = str(event.get("name", "unknown"))
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        ph = str(event.get("ph", ""))
        timestamps.append(ts)
        if dur > 0 and ph == "X":
            phase_totals[phase] += dur
            hot_events.append((dur, phase, name))
        event_counts[phase] += 1

    hot_events.sort(reverse=True)
    duration_us = 0.0
    if timestamps:
        duration_us = max(timestamps) - min(timestamps)
    hints: List[str] = []
    if phase_totals.get("wait", 0.0) > 0 and duration_us > 0 and phase_totals["wait"] / duration_us >= 0.2:
        hints.append("sync_heavy")
    if phase_totals.get("transfer", 0.0) > 0 and duration_us > 0 and phase_totals["transfer"] / duration_us >= 0.1:
        hints.append("transfer_heavy")
    if phase_totals.get("submit", 0.0) > 0 and event_counts.get("submit", 0) > 3:
        hints.append("multi_submit")

    return {
        "source_kind": "trace_event",
        "duration_us": duration_us,
        "phase_totals_us": dict(sorted(phase_totals.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "hot_events": [
            {"duration_us": dur, "phase": phase, "name": name}
            for dur, phase, name in hot_events[:10]
        ],
        "hints": hints,
    }


def iter_dicts(root: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(root, dict):
        yield root
        for value in root.values():
            yield from iter_dicts(value)
    elif isinstance(root, list):
        for value in root:
            yield from iter_dicts(value)


def summarize_generic_json(obj: Any) -> Dict[str, Any]:
    duration_candidates: List[Tuple[float, str]] = []
    name_counts: Dict[str, int] = defaultdict(int)
    for entry in iter_dicts(obj):
        name = None
        for key in ("name", "title", "symbol"):
            if key in entry and isinstance(entry[key], str):
                name = entry[key]
                break
        if name:
            name_counts[name] += 1
        for key in ("dur", "duration", "durationUs", "duration_ms", "weight"):
            if key in entry and isinstance(entry[key], (int, float)):
                duration_candidates.append((float(entry[key]), name or "unknown"))

    duration_candidates.sort(reverse=True)
    return {
        "source_kind": "json",
        "entry_count": sum(name_counts.values()),
        "name_counts": dict(sorted(name_counts.items())[:50]),
        "hot_entries": [
            {"value": value, "name": name}
            for value, name in duration_candidates[:10]
        ],
        "assumption": "Generic JSON summary uses best-effort name and duration fields; export a trace-event or xctrace JSON when possible.",
    }


def summarize_perf_stat(text: str) -> Dict[str, Any]:
    counters: Dict[str, float] = {}
    for line in text.splitlines():
        match = PERF_STAT_RE.match(line)
        if not match:
            continue
        counters[match.group(2)] = parse_number(match.group(1))

    hints: List[str] = []
    cycles = counters.get("cycles")
    instructions = counters.get("instructions")
    if cycles and instructions:
        ipc = instructions / cycles if cycles else 0.0
        if ipc < 1.0:
            hints.append("low_ipc")
    if counters.get("cache-misses", 0.0) > 0:
        hints.append("cache_miss_activity")

    return {
        "source_kind": "perf_stat",
        "counters": counters,
        "derived": {
            "ipc": (instructions / cycles) if cycles and instructions else 0.0,
        },
        "hints": hints,
    }


def summarize_perf_report(text: str) -> Dict[str, Any]:
    hotspots: List[Dict[str, Any]] = []
    for line in text.splitlines():
        match = PERF_REPORT_RE.match(line)
        if not match:
            continue
        hotspots.append({
            "percent": float(match.group(1)),
            "symbol": match.group(2),
        })
    return {
        "source_kind": "perf_report",
        "hotspots": hotspots[:20],
        "hints": ["cpu_hotspots_visible"] if hotspots else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize external profiling outputs into a unified JSON view.")
    parser.add_argument("--input", required=True, help="Path to a trace-event JSON, generic JSON, perf stat, or perf report text.")
    parser.add_argument("--kind", choices=["auto", "trace_event", "json", "perf_stat", "perf_report"], default="auto")
    parser.add_argument("--output", help="Optional path to write the summary JSON.")
    args = parser.parse_args()

    path = Path(args.input)
    text = load_text(path)
    kind = args.kind if args.kind != "auto" else detect_kind(path, text)

    if kind == "trace_event":
        summary = summarize_trace_events(load_json(path))
    elif kind == "json":
        summary = summarize_generic_json(load_json(path))
    elif kind == "perf_stat":
        summary = summarize_perf_stat(text)
    elif kind == "perf_report":
        summary = summarize_perf_report(text)
    else:
        raise RuntimeError(f"unsupported input kind: {kind}")

    summary["input"] = str(path)
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
