# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ov_contrib_license.adapters import (
    import_license_eye,
    import_ort,
    import_syft,
    run_syft,
)
from ov_contrib_license.adapters.common import ProviderError
from ov_contrib_license.discovery import DiscoveryOptions, RepositoryDiscovery
from ov_contrib_license.discovery.repository import DiscoveryError
from ov_contrib_license.model import (
    Decision,
    Inventory,
    InventoryBuilder,
    Provider,
    read_inventory,
)
from ov_contrib_license.policy import (
    apply_baseline,
    evaluate_inventory,
    load_baseline,
    load_policy,
)
from ov_contrib_license.reconciliation import (
    generate_tpp_preview,
    reconcile_artifact,
    reconcile_tpp,
)
from ov_contrib_license.reporters import (
    render_inventory,
    render_markdown,
    render_spdx,
    write_inventory,
)
from ov_contrib_license.toolchain import audit_toolchain

EXIT_SUCCESS = 0
EXIT_POLICY_FAIL = 2
EXIT_REVIEW_BLOCKING = 3
EXIT_CONFIGURATION_ERROR = 4
EXIT_DISCOVERY_ERROR = 5
EXIT_INTERNAL_ERROR = 6


class ConfigurationError(ValueError):
    pass


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ConfigurationError(message)


def _add_discovery_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-ref", help="base Git revision for incremental discovery"
    )
    parser.add_argument(
        "--head-ref", help="head Git revision for incremental discovery"
    )
    parser.add_argument(
        "--offline", action="store_true", help="forbid network providers"
    )
    parser.add_argument("--include", action="append", default=[], metavar="GLOB")
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB")


def _add_policy_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-dir", type=Path, help="policy directory")
    parser.add_argument("--baseline", type=Path, help="override baseline YAML")


def _add_provider_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ort-result", type=Path, help="import a previously generated ORT JSON result"
    )
    parser.add_argument(
        "--license-eye-result",
        type=Path,
        help="import a previously generated License Eye JSON result",
    )


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", type=Path, help="write the compliance report")
    parser.add_argument(
        "--report-format", choices=("markdown", "json"), default="markdown"
    )
    parser.add_argument(
        "--inventory-output", type=Path, help="write the canonical evaluated inventory"
    )
    parser.add_argument(
        "--github-summary",
        type=Path,
        help="append the Markdown report to a GitHub step-summary file",
    )
    parser.add_argument("--fail-on", choices=("REVIEW", "FAIL"), default="FAIL")


def build_parser() -> argparse.ArgumentParser:
    parser = ArgumentParser(
        prog="ov-contrib-license",
        description="OpenVINO contrib license compliance",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=ArgumentParser
    )

    inventory = subparsers.add_parser(
        "inventory", help="build a deterministic repository inventory"
    )
    inventory.add_argument("path", nargs="?", default=".", help="repository path")
    inventory.add_argument(
        "--output", type=Path, help="write the inventory to this file"
    )
    inventory.add_argument(
        "--format", dest="output_format", choices=("json", "yaml"), default="json"
    )
    _add_discovery_options(inventory)

    check = subparsers.add_parser(
        "check", help="evaluate repository components against policy"
    )
    check.add_argument("path", nargs="?", default=".", help="repository path")
    _add_discovery_options(check)
    _add_policy_options(check)
    _add_provider_options(check)
    _add_report_options(check)

    explain = subparsers.add_parser("explain", help="explain a component or finding")
    explain.add_argument("identifier")
    explain.add_argument("--path", default=".", help="repository path")
    explain.add_argument(
        "--inventory", type=Path, help="read an existing evaluated inventory"
    )
    _add_policy_options(explain)
    _add_provider_options(explain)

    tpp = subparsers.add_parser("tpp", help="reconcile third-party-programs.txt")
    tpp_commands = tpp.add_subparsers(
        dest="tpp_command", required=True, parser_class=ArgumentParser
    )
    tpp_check = tpp_commands.add_parser("check", help="check TPP coverage")
    tpp_check.add_argument("path", nargs="?", default=".", help="repository path")
    tpp_check.add_argument("--source-inventory", type=Path)
    tpp_check.add_argument("--tpp-file", type=Path, help="third-party programs file")
    _add_discovery_options(tpp_check)
    _add_policy_options(tpp_check)
    _add_provider_options(tpp_check)
    _add_report_options(tpp_check)
    tpp_generate = tpp_commands.add_parser(
        "generate", help="generate a non-authoritative TPP preview"
    )
    tpp_generate.add_argument("path", nargs="?", default=".", help="repository path")
    tpp_generate.add_argument("--source-inventory", type=Path)
    tpp_generate.add_argument("--tpp-file", type=Path)
    tpp_generate.add_argument("--output", type=Path, required=True)
    _add_policy_options(tpp_generate)

    sbom = subparsers.add_parser("sbom", help="generate a deterministic SPDX SBOM")
    sbom.add_argument("path", nargs="?", default=".", help="repository path")
    sbom.add_argument("--source-inventory", type=Path)
    sbom.add_argument("--format", choices=("spdx-json",), default="spdx-json")
    sbom.add_argument("--output", type=Path, required=True)
    _add_policy_options(sbom)
    _add_discovery_options(sbom)
    _add_provider_options(sbom)

    artifact = subparsers.add_parser(
        "artifact", help="verify a built artifact using Syft"
    )
    artifact_commands = artifact.add_subparsers(
        dest="artifact_command", required=True, parser_class=ArgumentParser
    )
    artifact_check = artifact_commands.add_parser(
        "check", help="reconcile artifact and source inventories"
    )
    artifact_check.add_argument("path", type=Path, help="install tree or package path")
    artifact_check.add_argument("--source-inventory", type=Path, required=True)
    artifact_check.add_argument(
        "--syft-result", type=Path, help="read Syft JSON instead of executing Syft"
    )
    _add_policy_options(artifact_check)
    _add_report_options(artifact_check)

    toolchain = subparsers.add_parser(
        "toolchain", help="audit compliance tooling and GitHub Actions"
    )
    toolchain_commands = toolchain.add_subparsers(
        dest="toolchain_command", required=True, parser_class=ArgumentParser
    )
    toolchain_check = toolchain_commands.add_parser(
        "check", help="run the toolchain audit"
    )
    toolchain_check.add_argument("path", nargs="?", default=".", help="repository path")
    _add_discovery_options(toolchain_check)
    _add_policy_options(toolchain_check)
    _add_report_options(toolchain_check)
    return parser


def _options(arguments: argparse.Namespace) -> DiscoveryOptions:
    return DiscoveryOptions(
        base_ref=getattr(arguments, "base_ref", None),
        head_ref=getattr(arguments, "head_ref", None),
        includes=tuple(getattr(arguments, "include", ())),
        excludes=tuple(getattr(arguments, "exclude", ())),
        offline=bool(getattr(arguments, "offline", False)),
    )


def _discover(path: str | Path, arguments: argparse.Namespace) -> Inventory:
    return RepositoryDiscovery(Path(path), _options(arguments)).run()


def _default_policy_dir(root: Path) -> Path:
    repository_policy = root / "tools" / "license_compliance" / "policy"
    if repository_policy.is_dir():
        return repository_policy
    return Path(__file__).resolve().parents[2] / "policy"


def _config(arguments: argparse.Namespace, root: Path):
    config = load_policy(arguments.policy_dir or _default_policy_dir(root))
    baseline_path = getattr(arguments, "baseline", None)
    if baseline_path:
        config = replace(config, baseline=load_baseline(baseline_path))
    return config


def _providers(
    inventory: Inventory, arguments: argparse.Namespace, config
) -> Inventory:
    ort_result = getattr(arguments, "ort_result", None)
    license_eye_result = getattr(arguments, "license_eye_result", None)
    if ort_result:
        inventory = import_ort(inventory, ort_result)
    if license_eye_result:
        inventory = import_license_eye(inventory, license_eye_result)
    builder = InventoryBuilder.from_inventory(inventory)
    present = {item.name for item in inventory.providers}
    for policy_name, provider_name in (("ort", "ort"), ("license_eye", "license-eye")):
        if provider_name in present:
            continue
        provider_config = config.providers.get(policy_name, {})
        version = (
            str(provider_config.get("version", "not-run"))
            if isinstance(provider_config, dict)
            else "not-run"
        )
        builder.add_provider(Provider(provider_name, version, "SKIPPED"))
    inventory = builder.build()
    return inventory


def _evaluated(root: Path, arguments: argparse.Namespace):
    config = _config(arguments, root)
    inventory = _providers(_discover(root, arguments), arguments, config)
    return evaluate_inventory(inventory, config).inventory, config


def _active_decisions(inventory: Inventory) -> tuple[int, int]:
    active = [item for item in inventory.findings if not item.suppressed]
    return (
        sum(item.decision is Decision.FAIL for item in active),
        sum(item.decision is Decision.REVIEW for item in active),
    )


def _exit_code(inventory: Inventory, fail_on: str) -> int:
    failures, reviews = _active_decisions(inventory)
    if failures:
        return EXIT_POLICY_FAIL
    if reviews and fail_on == "REVIEW":
        return EXIT_REVIEW_BLOCKING
    return EXIT_SUCCESS


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _report(
    inventory: Inventory,
    arguments: argparse.Namespace,
    *,
    title: str = "License Compliance",
) -> None:
    markdown = render_markdown(inventory, title)
    output_format = getattr(arguments, "report_format", "markdown")
    rendered = render_inventory(inventory) if output_format == "json" else markdown
    report_path = getattr(arguments, "report", None)
    if report_path:
        _write_text(report_path, rendered)
    else:
        print(rendered, end="")
    inventory_output = getattr(arguments, "inventory_output", None)
    if inventory_output:
        write_inventory(inventory, inventory_output)
    summary_path = getattr(arguments, "github_summary", None)
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as summary:
            summary.write(markdown)


def _run_inventory(arguments: argparse.Namespace) -> int:
    inventory = _discover(arguments.path, arguments)
    write_inventory(inventory, arguments.output, arguments.output_format)
    return EXIT_SUCCESS


def _run_check(arguments: argparse.Namespace) -> int:
    root = Path(arguments.path).resolve()
    inventory, _ = _evaluated(root, arguments)
    _report(inventory, arguments)
    return _exit_code(inventory, arguments.fail_on)


def _run_explain(arguments: argparse.Namespace) -> int:
    root = Path(arguments.path).resolve()
    inventory = (
        read_inventory(arguments.inventory)
        if arguments.inventory
        else _evaluated(root, arguments)[0]
    )
    component = next(
        (item for item in inventory.components if item.id == arguments.identifier), None
    )
    finding = next(
        (item for item in inventory.findings if item.id == arguments.identifier), None
    )
    if finding and component is None and finding.component_id:
        component = next(
            (item for item in inventory.components if item.id == finding.component_id),
            None,
        )
    related = [
        item
        for item in inventory.findings
        if item.id == arguments.identifier
        or (component and item.component_id == component.id)
    ]
    if component is None and finding is None:
        raise ConfigurationError(
            f"No component or finding matches {arguments.identifier!r}"
        )
    lines: list[str] = []
    if component:
        lines.extend(
            (
                f"Component: {component.id}",
                f"Name: {component.name}",
                f"Version: {component.version or 'unknown'}",
                f"Module: {component.module or 'repository'}",
                f"License: {component.declared_license or 'NOASSERTION'}",
                f"Relationship: {', '.join(item.value for item in component.relationships)}",
                f"Distribution: {component.distribution.value}",
                f"Obligations: {', '.join(item.value for item in component.obligations) or 'none'}",
            )
        )
    for item in related:
        lines.extend(
            (
                "",
                f"Finding: {item.id}",
                f"Code: {item.code}",
                f"Decision: {item.decision.value}",
                f"Suppressed: {'yes' if item.suppressed else 'no'}",
                f"Reason: {item.message}",
            )
        )
    print("\n".join(lines).rstrip())
    return EXIT_SUCCESS


def _run_tpp_check(arguments: argparse.Namespace) -> int:
    root = Path(arguments.path).resolve()
    if arguments.source_inventory:
        config = _config(arguments, root)
        source = read_inventory(arguments.source_inventory)
        source = replace(
            source,
            findings=tuple(
                item for item in source.findings if not item.code.startswith("POLICY_")
            ),
        )
        inventory = evaluate_inventory(source, config).inventory
    else:
        inventory, config = _evaluated(root, arguments)
    tpp_path = arguments.tpp_file or root / "third-party-programs.txt"
    inventory = apply_baseline(reconcile_tpp(inventory, tpp_path), config)
    _report(inventory, arguments, title="Third-Party Programs Reconciliation")
    return _exit_code(inventory, arguments.fail_on)


def _run_tpp_generate(arguments: argparse.Namespace) -> int:
    root = Path(arguments.path).resolve()
    inventory = (
        read_inventory(arguments.source_inventory)
        if arguments.source_inventory
        else _evaluated(root, arguments)[0]
    )
    existing = arguments.tpp_file or root / "third-party-programs.txt"
    _write_text(arguments.output, generate_tpp_preview(inventory, existing))
    return EXIT_SUCCESS


def _run_sbom(arguments: argparse.Namespace) -> int:
    root = Path(arguments.path).resolve()
    inventory = (
        read_inventory(arguments.source_inventory)
        if arguments.source_inventory
        else _evaluated(root, arguments)[0]
    )
    _write_text(arguments.output, render_spdx(inventory))
    return EXIT_SUCCESS


def _run_artifact_check(arguments: argparse.Namespace) -> int:
    source = read_inventory(arguments.source_inventory)
    root = Path.cwd().resolve()
    config = _config(arguments, root)
    if arguments.syft_result:
        artifact = import_syft(arguments.syft_result)
    else:
        syft_config = config.providers.get("syft", {})
        command = (
            str(syft_config.get("command", "syft"))
            if isinstance(syft_config, dict)
            else "syft"
        )
        artifact = run_syft(arguments.path, command)
    reconciled = reconcile_artifact(source, artifact)
    reconciled = replace(
        reconciled,
        findings=tuple(
            item for item in reconciled.findings if not item.code.startswith("POLICY_")
        ),
    )
    inventory = evaluate_inventory(reconciled, config).inventory
    _report(inventory, arguments, title="Artifact License Reconciliation")
    return _exit_code(inventory, arguments.fail_on)


def _run_toolchain_check(arguments: argparse.Namespace) -> int:
    root = Path(arguments.path).resolve()
    config = _config(arguments, root)
    result = audit_toolchain(root, _discover(root, arguments), config).inventory
    _report(result, arguments, title="License Toolchain Audit")
    return _exit_code(result, arguments.fail_on)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "inventory":
            return _run_inventory(arguments)
        if arguments.command == "check":
            return _run_check(arguments)
        if arguments.command == "explain":
            return _run_explain(arguments)
        if arguments.command == "tpp" and arguments.tpp_command == "check":
            return _run_tpp_check(arguments)
        if arguments.command == "tpp" and arguments.tpp_command == "generate":
            return _run_tpp_generate(arguments)
        if arguments.command == "sbom":
            return _run_sbom(arguments)
        if arguments.command == "artifact" and arguments.artifact_command == "check":
            return _run_artifact_check(arguments)
        if arguments.command == "toolchain" and arguments.toolchain_command == "check":
            return _run_toolchain_check(arguments)
        raise ConfigurationError(f"Unsupported command: {arguments.command}")
    except (ConfigurationError, ValueError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return EXIT_CONFIGURATION_ERROR
    except (DiscoveryError, ProviderError) as error:
        print(f"provider/discovery error: {error}", file=sys.stderr)
        return EXIT_DISCOVERY_ERROR
    except OSError as error:
        print(f"discovery I/O error: {error}", file=sys.stderr)
        return EXIT_DISCOVERY_ERROR
    except Exception as error:  # noqa: BLE001  # pragma: no cover - final CLI containment
        print(f"internal error: {error}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
