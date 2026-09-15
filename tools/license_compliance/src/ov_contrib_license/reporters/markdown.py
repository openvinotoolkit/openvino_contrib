# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ov_contrib_license.model import Decision, Inventory

MAX_FINDINGS_PER_DECISION = 20


def render_markdown(inventory: Inventory, title: str = "License Compliance") -> str:
    active = [item for item in inventory.findings if not item.suppressed]
    counts = {
        decision: sum(item.decision is decision for item in active)
        for decision in Decision
    }
    scopes = ", ".join(inventory.repository.impacted_scopes) or "full repository"
    lines = [
        f"# {title}",
        "",
        f"Impacted scopes: {scopes}",
        f"Components: {len(inventory.components)}",
        f"PASS: {counts[Decision.PASS]}",
        f"REVIEW: {counts[Decision.REVIEW]}",
        f"FAIL: {counts[Decision.FAIL]}",
        f"Suppressed by exact baseline: {sum(item.suppressed for item in inventory.findings)}",
    ]
    for decision in (Decision.FAIL, Decision.REVIEW):
        selected = [item for item in active if item.decision is decision]
        if not selected:
            continue
        lines.extend(("", f"## {decision.value}"))
        for finding in selected[:MAX_FINDINGS_PER_DECISION]:
            component = f" — `{finding.component_id}`" if finding.component_id else ""
            lines.extend(("", f"### {finding.code}{component}", "", finding.message))
            for remediation in finding.remediation:
                lines.append(f"- {remediation}")
        omitted = len(selected) - MAX_FINDINGS_PER_DECISION
        if omitted > 0:
            lines.extend(
                (
                    "",
                    f"_Omitted {omitted} additional {decision.value} findings; see the JSON inventory._",
                )
            )
    return "\n".join(lines).rstrip() + "\n"
