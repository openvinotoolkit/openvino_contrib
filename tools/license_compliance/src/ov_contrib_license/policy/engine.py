# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace

from ov_contrib_license.model import (
    Component,
    Decision,
    Finding,
    Inventory,
    Obligation,
    Relationship,
    Severity,
)

from .config import ExceptionRule, PolicyConfig, PolicyRule
from .expressions import (
    Expression,
    ExpressionError,
    parse_expression,
)

_DECISION_RANK = {Decision.PASS: 0, Decision.REVIEW: 1, Decision.FAIL: 2}


@dataclass(frozen=True)
class ComponentEvaluation:
    decision: Decision
    license_expression: str
    rule_ids: tuple[str, ...]
    obligations: tuple[Obligation, ...]
    exception_id: str | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    inventory: Inventory
    pass_count: int
    review_count: int
    fail_count: int
    suppressed_count: int

    @property
    def decision(self) -> Decision:
        if self.fail_count:
            return Decision.FAIL
        if self.review_count:
            return Decision.REVIEW
        return Decision.PASS


def _classify(license_id: str, config: PolicyConfig) -> str:
    base = license_id.split(" WITH ", 1)[0]
    for class_name, licenses in config.license_classes.items():
        if base in licenses:
            return class_name
    return "unknown"


def _rule_matches(
    rule: PolicyRule,
    *,
    license_expression: str,
    license_id: str,
    license_class: str,
    relationship: Relationship,
    component: Component,
) -> bool:
    if (
        rule.licenses
        and license_expression not in rule.licenses
        and license_id not in rule.licenses
    ):
        return False
    if rule.license_classes and license_class not in rule.license_classes:
        return False
    if rule.relationships and relationship not in rule.relationships:
        return False
    if rule.distributions and component.distribution not in rule.distributions:
        return False
    if rule.modules and (component.module or "repository") not in rule.modules:
        return False
    artifact_scope = dict(component.details).get("artifact_scope")
    return not rule.artifact_scopes or artifact_scope in rule.artifact_scopes


def _obligations(license_id: str, config: PolicyConfig) -> tuple[Obligation, ...]:
    result = set(config.obligations_by_class.get(_classify(license_id, config), ()))
    result.update(config.obligations_by_license.get(license_id, ()))
    return tuple(sorted(result, key=lambda item: item.value))


def _atomic_evaluation(
    license_expression: str,
    license_id: str,
    relationship: Relationship,
    component: Component,
    config: PolicyConfig,
) -> ComponentEvaluation:
    license_class = _classify(license_id, config)
    rule = next(
        (
            item
            for item in config.rules
            if _rule_matches(
                item,
                license_expression=license_expression,
                license_id=license_id,
                license_class=license_class,
                relationship=relationship,
                component=component,
            )
        ),
        None,
    )
    return ComponentEvaluation(
        decision=rule.decision if rule else Decision.REVIEW,
        license_expression=license_expression,
        rule_ids=(rule.id,) if rule else ("POLICY-NO-MATCH",),
        obligations=_obligations(license_id, config),
    )


def _combine(
    left: ComponentEvaluation,
    right: ComponentEvaluation,
    *,
    operator: str,
    expression: str,
) -> ComponentEvaluation:
    if operator == "OR":
        selected = min((left, right), key=lambda item: _DECISION_RANK[item.decision])
        obligations = selected.obligations
        rule_ids = selected.rule_ids
    else:
        selected = max((left, right), key=lambda item: _DECISION_RANK[item.decision])
        obligations = tuple(
            sorted(
                set(left.obligations + right.obligations), key=lambda item: item.value
            )
        )
        rule_ids = tuple(sorted(set(left.rule_ids + right.rule_ids)))
    return ComponentEvaluation(
        decision=selected.decision,
        license_expression=expression,
        rule_ids=rule_ids,
        obligations=obligations,
    )


def _evaluate_expression(
    expression: Expression,
    relationship: Relationship,
    component: Component,
    config: PolicyConfig,
) -> ComponentEvaluation:
    rendered = expression.render()
    exact = [
        rule for rule in config.rules if rule.licenses and rendered in rule.licenses
    ]
    for rule in exact:
        if _rule_matches(
            rule,
            license_expression=rendered,
            license_id=rendered,
            license_class=_classify(rendered, config),
            relationship=relationship,
            component=component,
        ):
            return ComponentEvaluation(
                rule.decision,
                rendered,
                (rule.id,),
                _obligations(rendered, config),
            )
    if expression.operator == "LICENSE":
        assert expression.value is not None
        return _atomic_evaluation(
            rendered, expression.value, relationship, component, config
        )
    assert expression.left is not None and expression.right is not None
    if expression.operator == "WITH":
        return _atomic_evaluation(
            rendered,
            rendered,
            relationship,
            component,
            config,
        )
    left = _evaluate_expression(expression.left, relationship, component, config)
    right = _evaluate_expression(expression.right, relationship, component, config)
    return _combine(left, right, operator=expression.operator, expression=rendered)


def _component_license(component: Component) -> str:
    if component.declared_license:
        return component.declared_license
    if component.detected_licenses:
        return " AND ".join(component.detected_licenses)
    return "NOASSERTION"


def _matching_exception(
    component: Component,
    config: PolicyConfig,
    today: dt.date,
) -> tuple[ExceptionRule | None, ExceptionRule | None, str | None]:
    exact = [item for item in config.exceptions if item.component == component.id]
    mismatch: tuple[ExceptionRule, str] | None = None
    for item in exact:
        if item.expires < today:
            mismatch = mismatch or (item, "POLICY_EXCEPTION_EXPIRED")
            continue
        if item.module != (component.module or "repository"):
            mismatch = mismatch or (item, "POLICY_EXCEPTION_MODULE_MISMATCH")
            continue
        if component.distribution not in item.distributions:
            mismatch = mismatch or (item, "POLICY_EXCEPTION_DISTRIBUTION_MISMATCH")
            continue
        if not set(component.relationships).issubset(item.relationships):
            mismatch = mismatch or (item, "POLICY_EXCEPTION_RELATIONSHIP_MISMATCH")
            continue
        return item, None, None
    if mismatch:
        return None, mismatch[0], mismatch[1]
    identity = component.id.rsplit("@", 1)[0]
    version_mismatch = next(
        (
            item
            for item in config.exceptions
            if item.component.rsplit("@", 1)[0] == identity
        ),
        None,
    )
    if version_mismatch:
        return None, version_mismatch, "POLICY_EXCEPTION_VERSION_MISMATCH"
    return None, None, None


def _evaluate_component(
    component: Component,
    config: PolicyConfig,
    today: dt.date,
) -> tuple[Component, Finding, str | None]:
    raw_expression = _component_license(component)
    try:
        expression = parse_expression(raw_expression)
        normalized = expression.render()
    except ExpressionError as error:
        finding = Finding.create(
            code="POLICY_INVALID_LICENSE_EXPRESSION",
            severity=Severity.ERROR,
            decision=Decision.REVIEW,
            component_id=component.id,
            message=f"Invalid SPDX license expression {raw_expression!r}: {error}",
            evidence=component.evidence,
            remediation=("Replace the value with a valid SPDX license expression.",),
            fingerprint_values=(
                raw_expression,
                *component.relationships,
                component.distribution,
                component.module or "",
            ),
        )
        return component, finding, None

    evaluations = [
        _evaluate_expression(expression, relationship, component, config)
        for relationship in component.relationships
    ]
    evaluation = max(evaluations, key=lambda item: _DECISION_RANK[item.decision])
    obligations = tuple(
        sorted(
            {obligation for item in evaluations for obligation in item.obligations},
            key=lambda item: item.value,
        )
    )
    updated = replace(component, declared_license=normalized, obligations=obligations)
    exception, exception_issue, issue_code = _matching_exception(updated, config, today)
    if exception_issue:
        issue_decision = (
            Decision.FAIL
            if issue_code == "POLICY_EXCEPTION_EXPIRED"
            else max(
                (evaluation.decision, Decision.REVIEW),
                key=lambda item: _DECISION_RANK[item],
            )
        )
        issue_description = (
            issue_code.removeprefix("POLICY_EXCEPTION_").replace("_", " ").lower()
        )
        if issue_code == "POLICY_EXCEPTION_EXPIRED":
            issue_description += f" on {exception_issue.expires.isoformat()}"
        finding = Finding.create(
            code=issue_code or "POLICY_EXCEPTION_MISMATCH",
            severity=Severity.ERROR,
            decision=issue_decision,
            component_id=component.id,
            message=(
                f"Policy exception {exception_issue.id} has {issue_description} for "
                f"component {component.id}."
            ),
            evidence=component.evidence,
            remediation=(
                "Update the narrowly approved exception or resolve the underlying finding.",
            ),
            fingerprint_values=(
                exception_issue.id,
                exception_issue.component,
                exception_issue.expires.isoformat(),
                component.id,
                *(item.value for item in component.relationships),
                component.distribution.value,
                component.module or "repository",
            ),
        )
        return updated, finding, exception_issue.id
    if exception:
        evaluation = replace(
            evaluation,
            decision=exception.decision,
            rule_ids=(f"exception:{exception.id}",),
            exception_id=exception.id,
        )

    code = {
        Decision.PASS: "POLICY_COMPONENT_PASS",
        Decision.REVIEW: "POLICY_COMPONENT_REVIEW",
        Decision.FAIL: "POLICY_COMPONENT_FAIL",
    }[evaluation.decision]
    severity = {
        Decision.PASS: Severity.INFO,
        Decision.REVIEW: Severity.WARNING,
        Decision.FAIL: Severity.ERROR,
    }[evaluation.decision]
    rules = ", ".join(evaluation.rule_ids)
    finding = Finding.create(
        code=code,
        severity=severity,
        decision=evaluation.decision,
        component_id=component.id,
        message=(
            f"{component.name}: {normalized}; relationship="
            f"{','.join(item.value for item in component.relationships)}; "
            f"distribution={component.distribution.value}; rule={rules}."
        ),
        evidence=component.evidence,
        remediation=(
            "Add stronger license/distribution evidence or a narrowly approved exception."
            if evaluation.decision is not Decision.PASS
            else "No policy action required.",
        ),
        fingerprint_values=(
            *evaluation.rule_ids,
            normalized,
            *(item.value for item in component.relationships),
            component.distribution.value,
            component.module or "repository",
        ),
    )
    return updated, finding, exception.id if exception else None


def evaluate_inventory(
    inventory: Inventory,
    config: PolicyConfig,
    *,
    today: dt.date | None = None,
) -> EvaluationSummary:
    today = today or dt.datetime.now(dt.timezone.utc).date()
    components: list[Component] = []
    findings = list(inventory.findings)
    used_exceptions: set[str] = set()
    for component in inventory.components:
        updated, finding, exception_id = _evaluate_component(component, config, today)
        components.append(updated)
        findings.append(finding)
        if exception_id:
            used_exceptions.add(exception_id)

    if inventory.repository.base_ref is None:
        for exception in config.exceptions:
            if exception.id in used_exceptions:
                continue
            findings.append(
                Finding.create(
                    code="POLICY_EXCEPTION_UNUSED",
                    severity=Severity.WARNING,
                    decision=Decision.REVIEW,
                    component_id=exception.component,
                    message=f"Policy exception {exception.id} did not match any component.",
                    remediation=(
                        "Remove the stale exception or correct its exact scope.",
                    ),
                    fingerprint_values=(exception.id, exception.component),
                )
            )

    baseline = config.baseline_by_id
    evaluated_findings = tuple(
        sorted(
            (
                finding.suppress(baseline[finding.id].reason)
                if finding.id in baseline
                else finding
                for finding in findings
            ),
            key=Finding.sort_key,
        )
    )
    evaluated = replace(
        inventory,
        components=tuple(sorted(components, key=lambda item: item.id)),
        findings=evaluated_findings,
    )
    active = [item for item in evaluated_findings if not item.suppressed]
    return EvaluationSummary(
        inventory=evaluated,
        pass_count=sum(item.decision is Decision.PASS for item in active),
        review_count=sum(item.decision is Decision.REVIEW for item in active),
        fail_count=sum(item.decision is Decision.FAIL for item in active),
        suppressed_count=sum(item.suppressed for item in evaluated_findings),
    )


def apply_baseline(inventory: Inventory, config: PolicyConfig) -> Inventory:
    baseline = config.baseline_by_id
    return replace(
        inventory,
        findings=tuple(
            sorted(
                (
                    item.suppress(baseline[item.id].reason)
                    if item.id in baseline and not item.suppressed
                    else item
                    for item in inventory.findings
                ),
                key=Finding.sort_key,
            )
        ),
    )
