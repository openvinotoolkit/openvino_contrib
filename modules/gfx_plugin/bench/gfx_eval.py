#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


GLOBAL_RE = re.compile(r"GLOBAL max_abs_diff=([0-9eE+.\-]+) max_rel_diff=([0-9eE+.\-]+)")
REFERENCE_RE = re.compile(r"REFERENCE device=(.+)")
GFX_ONLY_RE = re.compile(r"^GFX_ONLY$", re.MULTILINE)
PER_OP_MATCH_RE = re.compile(r"^PER_OP_MATCH$", re.MULTILINE)
PER_OP_RE = re.compile(
    r"^\[op (\d+)\] (.+?) \(([^()]+)\) max_abs_diff=([0-9eE+.\-]+) max_rel_diff=([0-9eE+.\-]+)$",
    re.MULTILINE,
)
GFX_OUTPUT_RE = re.compile(
    r"^(\S+)\s+elements=(\d+)\s+finite=(\d+)\s+nan=(\d+)\s+inf=(\d+)\s+min=([0-9eE+.\-]+)\s+max=([0-9eE+.\-]+)\s+mean=([0-9eE+.\-]+)\s+l2=([0-9eE+.\-]+)$",
    re.MULTILINE,
)


@dataclass
class RunResult:
    command: List[str]
    stdout: str


def log(message: str) -> None:
    print(f"gfx_eval: {message}", file=sys.stderr, flush=True)


def sh_join(argv: List[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def redact_command(argv: List[str]) -> List[str]:
    redacted: List[str] = []
    hide_next = False
    for arg in argv:
        if hide_next:
            redacted.append("********")
            hide_next = False
            continue
        redacted.append(arg)
        if arg == "-p" and redacted and redacted[0] == "sshpass":
            hide_next = True
    return redacted


def run_subprocess(argv: List[str],
                   *,
                   env: Optional[Dict[str, str]] = None,
                   stream_output: bool = False,
                   error_prefix: str = "command") -> str:
    log(f"run: {sh_join(redact_command(argv))}")
    proc = subprocess.Popen(argv,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=env)
    stdout_chunks: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_chunks.append(line)
        if stream_output:
            print(line, end="", file=sys.stderr, flush=True)
    return_code = proc.wait()
    stdout = "".join(stdout_chunks)
    if return_code != 0:
        raise RuntimeError(f"{error_prefix} failed ({return_code}): {sh_join(redact_command(argv))}\n{stdout}")
    return stdout


def resolve_existing_path(path_str: str) -> Optional[Path]:
    path = Path(path_str)
    if path.exists():
        return path.resolve()
    return None


def stat_fingerprint(path: Path) -> Dict[str, object]:
    if path.is_symlink():
        return {
            "type": "symlink",
            "target": os.readlink(path),
        }
    stat = path.stat()
    return {
        "type": "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def snapshot_directory(path: Path) -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for child in sorted(path.rglob("*")):
        if child.is_dir() and not child.is_symlink():
            continue
        snapshot[str(child.relative_to(path))] = stat_fingerprint(child)
    return snapshot


def is_runtime_library_entry(path: Path) -> bool:
    name = path.name
    if path.is_symlink():
        return True
    if not path.is_file():
        return False
    return ".so" in name or name.endswith(".dylib")


def snapshot_runtime_directory(path: Path) -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for child in sorted(path.rglob("*")):
        if child.is_dir() and not child.is_symlink():
            continue
        if not is_runtime_library_entry(child):
            continue
        snapshot[str(child.relative_to(path))] = stat_fingerprint(child)
    return snapshot


def parse_compare_output(stdout: str) -> Dict[str, object]:
    per_op = []
    for match in PER_OP_RE.finditer(stdout):
        per_op.append({
            "index": int(match.group(1)),
            "name": match.group(2),
            "type": match.group(3),
            "max_abs_diff": float(match.group(4)),
            "max_rel_diff": float(match.group(5)),
        })

    if GFX_ONLY_RE.search(stdout):
        outputs = []
        for match in GFX_OUTPUT_RE.finditer(stdout):
            outputs.append({
                "name": match.group(1),
                "elements": int(match.group(2)),
                "finite": int(match.group(3)),
                "nan": int(match.group(4)),
                "inf": int(match.group(5)),
                "min": float(match.group(6)),
                "max": float(match.group(7)),
                "mean": float(match.group(8)),
                "l2": float(match.group(9)),
            })
        if not outputs:
            raise RuntimeError("failed to parse gfx-only ov_gfx_compare_runner output")
        return {
            "mode": "gfx_only",
            "outputs": outputs,
        }

    global_match = GLOBAL_RE.search(stdout)
    reference_match = REFERENCE_RE.search(stdout)
    if per_op and not global_match and PER_OP_MATCH_RE.search(stdout):
        return {
            "mode": "per_op",
            "per_op": per_op,
            "max_abs_diff": max(item["max_abs_diff"] for item in per_op),
            "max_rel_diff": max(item["max_rel_diff"] for item in per_op),
        }
    if not global_match:
        raise RuntimeError("failed to parse ov_gfx_compare_runner output")
    result: Dict[str, object] = {
        "max_abs_diff": float(global_match.group(1)),
        "max_rel_diff": float(global_match.group(2)),
    }
    if per_op:
        result["per_op"] = per_op
    if reference_match:
        result["reference_device"] = reference_match.group(1).strip()
    return result


def discover_android_ndk_shared_lib() -> Optional[Path]:
    candidate_roots: List[Path] = []
    for env_key in ("ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME", "NDK_ROOT"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidate_roots.append(Path(env_value))

    sdk_root = os.environ.get("ANDROID_SDK_ROOT") or os.environ.get("ANDROID_HOME")
    if sdk_root:
        candidate_roots.extend(sorted((Path(sdk_root) / "ndk").glob("*"), reverse=True))

    candidate_roots.extend(sorted((Path.home() / "Library/Android/sdk/ndk").glob("*"), reverse=True))

    seen: set[Path] = set()
    for root in candidate_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / "toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
        if candidate.exists():
            return candidate.resolve()
    return None


def discover_sibling_artifact(reference_path: str, artifact_name: str) -> Optional[Path]:
    reference = resolve_existing_path(reference_path)
    if not reference:
        return None
    candidate = reference.parent / artifact_name
    if candidate.exists():
        return candidate.resolve()
    return None


def append_runtime_file_if_missing(runtime_files: List[str], candidate: Optional[Path]) -> None:
    if not candidate:
        return
    if any(Path(item).name == candidate.name for item in runtime_files):
        return
    runtime_files.append(str(candidate))


def strip_transport_warnings(payload: str) -> str:
    cleaned_lines = []
    for line in payload.splitlines():
        if line.startswith("Warning: Permanently added "):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def discover_reference_plugin(plugin_lib: str, reference_device: str) -> Optional[Path]:
    if reference_device == "TEMPLATE":
        return discover_sibling_artifact(plugin_lib, "libopenvino_template_plugin.so")
    return None


class ExecContext:
    def prepare_runtime_path(self, path_str: str, executable: bool = False) -> str:
        raise NotImplementedError

    def prepare_lib_dir(self, path_str: str) -> str:
        raise NotImplementedError

    def run(self, argv: List[str], env: Dict[str, str]) -> RunResult:
        raise NotImplementedError

    def make_report_dir(self, name: str) -> str:
        raise NotImplementedError

    def read_text_file(self, path_str: str) -> str:
        raise NotImplementedError

    def cleanup(self) -> None:
        pass


class HostExecContext(ExecContext):
    def __init__(self):
        self._tempdirs: List[tempfile.TemporaryDirectory[str]] = []

    def prepare_runtime_path(self, path_str: str, executable: bool = False) -> str:
        path = resolve_existing_path(path_str)
        return str(path if path else path_str)

    def prepare_lib_dir(self, path_str: str) -> str:
        path = resolve_existing_path(path_str)
        return str(path if path else path_str)

    def run(self, argv: List[str], env: Dict[str, str]) -> RunResult:
        merged_env = os.environ.copy()
        merged_env.update(env)
        stdout = run_subprocess(argv, env=merged_env, stream_output=True)
        return RunResult(command=argv, stdout=stdout)

    def make_report_dir(self, name: str) -> str:
        tempdir = tempfile.TemporaryDirectory(prefix=f"{name}_")
        self._tempdirs.append(tempdir)
        return tempdir.name

    def read_text_file(self, path_str: str) -> str:
        return Path(path_str).read_text(encoding="utf-8")

    def cleanup(self) -> None:
        for tempdir in self._tempdirs:
            tempdir.cleanup()
        self._tempdirs.clear()


class AndroidExecContext(ExecContext):
    def __init__(self, remote_dir: str, adb_serial: Optional[str]):
        self.remote_dir = remote_dir.rstrip("/")
        self.adb_serial = adb_serial
        self._created = False
        self._manifest_loaded = False
        self._manifest: Dict[str, Dict[str, Any]] = {"files": {}, "dirs": {}}
        self._pushed_paths: Dict[str, str] = {}
        self._remote_lib_dirs: Dict[str, str] = {}

    def _adb(self, args: List[str], stream_output: bool = False) -> str:
        cmd = ["adb"]
        if self.adb_serial:
            cmd.extend(["-s", self.adb_serial])
        cmd.extend(args)
        return run_subprocess(cmd, stream_output=stream_output, error_prefix="adb command")

    def _ensure_remote_dir(self) -> None:
        if self._created:
            if not self._manifest_loaded:
                self._load_manifest()
            return
        self._adb(["shell", "mkdir", "-p", self.remote_dir])
        self._adb(["shell", "mkdir", "-p", f"{self.remote_dir}/libs"])
        self._created = True
        self._load_manifest()

    def _manifest_path(self) -> str:
        return f"{self.remote_dir}/.gfx_eval_manifest.json"

    def _load_manifest(self) -> None:
        if self._manifest_loaded:
            return
        payload = strip_transport_warnings(
            self._adb(["shell", f"if [ -f {shlex.quote(self._manifest_path())} ]; then cat {shlex.quote(self._manifest_path())}; fi"]))
        if payload.strip():
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    self._manifest["files"] = dict(data.get("files", {}))
                    self._manifest["dirs"] = dict(data.get("dirs", {}))
            except json.JSONDecodeError:
                log("manifest is invalid on android target; restaging from scratch in-place")
        self._manifest_loaded = True

    def _save_manifest(self) -> None:
        self._ensure_remote_dir()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            json.dump(self._manifest, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = tmp.name
        try:
            self._adb(["push", tmp_path, self._manifest_path()])
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _upload_single_path(self, local_path: Path, remote_path: str, executable: bool = False) -> None:
        if local_path.is_symlink():
            target = os.readlink(local_path)
            self._adb(["shell", "ln", "-sfn", target, remote_path])
        else:
            self._adb(["push", str(local_path), remote_path])
            if executable:
                self._adb(["shell", "chmod", "755", remote_path])

    def _remove_remote_path(self, remote_path: str) -> None:
        self._adb(["shell", "rm", "-rf", remote_path])

    def prepare_runtime_path(self, path_str: str, executable: bool = False) -> str:
        local_path = resolve_existing_path(path_str)
        if not local_path:
            return path_str
        if str(local_path) in self._pushed_paths:
            return self._pushed_paths[str(local_path)]
        self._ensure_remote_dir()
        remote_path = f"{self.remote_dir}/{local_path.name}"
        key = str(local_path)
        fingerprint = stat_fingerprint(local_path)
        if self._manifest["files"].get(key) == fingerprint:
            log(f"reuse file: {local_path} -> {remote_path}")
        else:
            log(f"stage file: {local_path} -> {remote_path}")
            self._upload_single_path(local_path, remote_path, executable=executable)
            self._manifest["files"][key] = fingerprint
            self._save_manifest()
        self._pushed_paths[key] = remote_path
        return remote_path

    def prepare_lib_dir(self, path_str: str) -> str:
        local_path = resolve_existing_path(path_str)
        if not local_path:
            return path_str
        key = str(local_path)
        if key in self._remote_lib_dirs:
            return self._remote_lib_dirs[key]
        self._ensure_remote_dir()
        remote_path = f"{self.remote_dir}/libs/{local_path.name}"
        new_snapshot = snapshot_runtime_directory(local_path)
        old_snapshot = self._manifest["dirs"].get(key, {})
        removed = sorted(set(old_snapshot.keys()) - set(new_snapshot.keys()), reverse=True)
        changed = [rel for rel, fp in new_snapshot.items() if old_snapshot.get(rel) != fp]
        if not removed and not changed:
            log(f"reuse lib dir: {local_path} -> {remote_path}")
        else:
            log(f"sync lib dir: {local_path} -> {remote_path} changed={len(changed)} removed={len(removed)}")
            if removed and len(removed) > 32:
                self._adb(["shell", "rm", "-rf", remote_path])
                self._adb(["shell", "mkdir", "-p", remote_path])
                changed = sorted(new_snapshot.keys())
                removed = []
            else:
                self._adb(["shell", "mkdir", "-p", remote_path])
                parent_dirs = sorted({str(Path(f"{remote_path}/{rel}").parent).replace("\\", "/") for rel in changed})
                if parent_dirs:
                    self._adb(["shell", "mkdir", "-p", *parent_dirs])
                for rel in removed:
                    self._remove_remote_path(f"{remote_path}/{rel}")
            for rel in changed:
                self._upload_single_path(local_path / rel, f"{remote_path}/{rel}")
            self._manifest["dirs"][key] = new_snapshot
            self._save_manifest()
        self._remote_lib_dirs[key] = remote_path
        return remote_path

    def run(self, argv: List[str], env: Dict[str, str]) -> RunResult:
        self._ensure_remote_dir()
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        shell_cmd = f"cd {shlex.quote(self.remote_dir)} && {exports} {sh_join(argv)}"
        stdout = self._adb(["shell", shell_cmd], stream_output=True)
        return RunResult(command=["adb", "shell", shell_cmd], stdout=stdout)

    def make_report_dir(self, name: str) -> str:
        self._ensure_remote_dir()
        remote_path = f"{self.remote_dir}/{name}"
        self._adb(["shell", "rm", "-rf", remote_path])
        self._adb(["shell", "mkdir", "-p", remote_path])
        return remote_path

    def read_text_file(self, path_str: str) -> str:
        return self._adb(["shell", "cat", path_str])

    def cleanup(self) -> None:
        if self._created:
            self._adb(["shell", "rm", "-rf", self.remote_dir])


def parse_device_file(path_str: str) -> Dict[str, str]:
    path = Path(path_str)
    payload: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        payload[key.strip()] = value.strip().replace("\\ ", " ")
    return payload


class SshExecContext(ExecContext):
    def __init__(self, host: str, user: str, password: str, remote_dir: str):
        self.host = host
        self.user = user
        self.password = password
        self.remote_dir = remote_dir.rstrip("/")
        self._created = False
        self._manifest_loaded = False
        self._manifest: Dict[str, Dict[str, Any]] = {"files": {}, "dirs": {}}
        self._pushed_paths: Dict[str, str] = {}
        self._remote_lib_dirs: Dict[str, str] = {}

    def _sshpass_prefix(self) -> List[str]:
        return ["sshpass", "-p", self.password]

    def _ssh(self, remote_command: str, stream_output: bool = False) -> str:
        cmd = self._sshpass_prefix() + [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            f"{self.user}@{self.host}",
            remote_command,
        ]
        return run_subprocess(cmd, stream_output=stream_output, error_prefix="ssh command")

    def _scp(self, sources: List[str], destination: str, recursive: bool = False) -> None:
        cmd = self._sshpass_prefix() + ["scp"]
        if recursive:
            cmd.append("-r")
        cmd.extend([
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
        ])
        cmd.extend(sources)
        cmd.append(f"{self.user}@{self.host}:{destination}")
        run_subprocess(cmd, error_prefix="scp command")

    def _ensure_remote_dir(self) -> None:
        if self._created:
            if not self._manifest_loaded:
                self._load_manifest()
            return
        self._ssh(f"mkdir -p {shlex.quote(self.remote_dir)} {shlex.quote(self.remote_dir + '/libs')}")
        self._created = True
        self._load_manifest()

    def _manifest_path(self) -> str:
        return f"{self.remote_dir}/.gfx_eval_manifest.json"

    def _load_manifest(self) -> None:
        if self._manifest_loaded:
            return
        payload = strip_transport_warnings(
            self._ssh(f"if [ -f {shlex.quote(self._manifest_path())} ]; then cat {shlex.quote(self._manifest_path())}; fi"))
        if payload.strip():
            try:
                data = json.loads(payload)
                if isinstance(data, dict):
                    self._manifest["files"] = dict(data.get("files", {}))
                    self._manifest["dirs"] = dict(data.get("dirs", {}))
            except json.JSONDecodeError:
                log("manifest is invalid on ssh target; restaging from scratch in-place")
        self._manifest_loaded = True

    def _save_manifest(self) -> None:
        self._ensure_remote_dir()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            json.dump(self._manifest, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            tmp_path = tmp.name
        try:
            self._scp([tmp_path], self._manifest_path())
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _upload_single_path(self, local_path: Path, remote_path: str, executable: bool = False) -> None:
        if local_path.is_symlink():
            target = os.readlink(local_path)
            self._ssh(f"ln -sfn {shlex.quote(target)} {shlex.quote(remote_path)}")
        else:
            self._scp([str(local_path)], remote_path)
            if executable:
                self._ssh(f"chmod 755 {shlex.quote(remote_path)}")

    def _remove_remote_path(self, remote_path: str) -> None:
        self._ssh(f"rm -rf {shlex.quote(remote_path)}")

    def prepare_runtime_path(self, path_str: str, executable: bool = False) -> str:
        local_path = resolve_existing_path(path_str)
        if not local_path:
            return path_str
        key = str(local_path)
        if key in self._pushed_paths:
            return self._pushed_paths[key]
        self._ensure_remote_dir()
        remote_path = f"{self.remote_dir}/{local_path.name}"
        fingerprint = stat_fingerprint(local_path)
        if self._manifest["files"].get(key) == fingerprint:
            log(f"reuse file: {local_path} -> {self.user}@{self.host}:{remote_path}")
        else:
            log(f"stage file: {local_path} -> {self.user}@{self.host}:{remote_path}")
            self._upload_single_path(local_path, remote_path, executable=executable)
            self._manifest["files"][key] = fingerprint
            self._save_manifest()
        self._pushed_paths[key] = remote_path
        return remote_path

    def prepare_lib_dir(self, path_str: str) -> str:
        local_path = resolve_existing_path(path_str)
        if not local_path:
            return path_str
        key = str(local_path)
        if key in self._remote_lib_dirs:
            return self._remote_lib_dirs[key]
        self._ensure_remote_dir()
        remote_path = f"{self.remote_dir}/libs/{local_path.name}"
        new_snapshot = snapshot_runtime_directory(local_path)
        old_snapshot = self._manifest["dirs"].get(key, {})
        removed = sorted(set(old_snapshot.keys()) - set(new_snapshot.keys()), reverse=True)
        changed = [rel for rel, fp in new_snapshot.items() if old_snapshot.get(rel) != fp]
        if not removed and not changed:
            log(f"reuse lib dir: {local_path} -> {self.user}@{self.host}:{remote_path}")
        else:
            log(f"sync lib dir: {local_path} -> {self.user}@{self.host}:{remote_path} changed={len(changed)} removed={len(removed)}")
            if removed and len(removed) > 32:
                self._ssh(f"rm -rf {shlex.quote(remote_path)} && mkdir -p {shlex.quote(remote_path)}")
                changed = sorted(new_snapshot.keys())
                removed = []
            else:
                self._ssh(f"mkdir -p {shlex.quote(remote_path)}")
                parent_dirs = sorted({str(Path(f"{remote_path}/{rel}").parent).replace('\\', '/') for rel in changed})
                if parent_dirs:
                    self._ssh("mkdir -p " + " ".join(shlex.quote(parent) for parent in parent_dirs))
                for rel in removed:
                    self._remove_remote_path(f"{remote_path}/{rel}")
            for rel in changed:
                self._upload_single_path(local_path / rel, f"{remote_path}/{rel}")
            self._manifest["dirs"][key] = new_snapshot
            self._save_manifest()
        self._remote_lib_dirs[key] = remote_path
        return remote_path

    def run(self, argv: List[str], env: Dict[str, str]) -> RunResult:
        self._ensure_remote_dir()
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        shell_cmd = f"cd {shlex.quote(self.remote_dir)} && {exports} {sh_join(argv)}"
        stdout = self._ssh(shell_cmd, stream_output=True)
        return RunResult(command=["ssh", f"{self.user}@{self.host}", shell_cmd], stdout=stdout)

    def make_report_dir(self, name: str) -> str:
        self._ensure_remote_dir()
        remote_path = f"{self.remote_dir}/{name}"
        self._ssh(f"rm -rf {shlex.quote(remote_path)} && mkdir -p {shlex.quote(remote_path)}")
        return remote_path

    def read_text_file(self, path_str: str) -> str:
        return self._ssh(f"cat {shlex.quote(path_str)}")

    def cleanup(self) -> None:
        if self._created:
            self._ssh(f"rm -rf {shlex.quote(self.remote_dir)}")


def make_runtime_env(plugin_lib: str,
                     lib_dirs: List[str],
                     runtime_files: List[str],
                     target_platform: str) -> Dict[str, str]:
    env = {
        "GFX_PLUGIN_PATH": plugin_lib,
    }
    search_dirs: List[str] = []
    for candidate in [plugin_lib, *runtime_files]:
        parent = str(Path(candidate).parent)
        if parent and parent not in search_dirs:
            search_dirs.append(parent)
    for lib_dir in lib_dirs:
        if lib_dir not in search_dirs:
            search_dirs.append(lib_dir)
    if search_dirs:
        lib_var = "DYLD_LIBRARY_PATH" if target_platform == "host" and platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        env[lib_var] = ":".join(search_dirs)
    return env


def add_model_companion_files(ctx: ExecContext, model_path: str) -> str:
    model = resolve_existing_path(model_path)
    if not model:
        return model_path
    xml_remote = ctx.prepare_runtime_path(str(model))
    bin_path = model.with_suffix(".bin")
    if bin_path.exists():
        ctx.prepare_runtime_path(str(bin_path))
    return xml_remote


def run_compare(ctx: ExecContext,
                args: argparse.Namespace,
                runtime_env: Dict[str, str]) -> Dict[str, object]:
    compare_runner = ctx.prepare_runtime_path(args.compare_runner, executable=True)
    model_path = add_model_companion_files(ctx, args.model)
    cmd = [compare_runner, model_path]
    if args.gfx_only:
        cmd.append("--gfx-only")
    else:
        cmd.extend(["--reference-device", args.reference_device])
    if args.per_op_all:
        cmd.append("--per-op-all")
    if args.reference_plugin and not args.gfx_only:
        reference_plugin = ctx.prepare_runtime_path(args.reference_plugin)
        cmd.extend(["--reference-plugin", reference_plugin])
    cmd.extend(args.compare_arg)
    log(f"compare mode={'gfx_only' if args.gfx_only else args.reference_device + '_vs_GFX'}")
    result = ctx.run(cmd, runtime_env)
    metrics = parse_compare_output(result.stdout)
    metrics["command"] = sh_join(result.command)
    metrics["raw_output"] = result.stdout
    return metrics


def prepare_runtime_files(ctx: ExecContext, runtime_files: List[str]) -> List[str]:
    prepared: List[str] = []
    for path_str in runtime_files:
        prepared.append(ctx.prepare_runtime_path(path_str))
    return prepared


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GFX accuracy-only comparison on host, Android, or SSH targets.")
    parser.add_argument("--platform", choices=("host", "android", "ssh"), required=True)
    parser.add_argument("--model", required=True, help="Path to model .xml")
    parser.add_argument("--compare-runner", required=True, help="Path to ov_gfx_compare_runner")
    parser.add_argument("--plugin-lib", required=True, help="Path to openvino_gfx_plugin library")
    parser.add_argument("--lib-dir", action="append", default=[],
                        help="Runtime library directory to add to LD_LIBRARY_PATH/DYLD_LIBRARY_PATH")
    parser.add_argument("--runtime-file", action="append", default=[],
                        help="Extra runtime file to stage next to the binaries. Repeat as needed.")
    parser.add_argument("--reference-device", default="TEMPLATE",
                        help="Reference device for accuracy compare, for example TEMPLATE")
    parser.add_argument("--reference-plugin",
                        help="Optional explicit reference plugin library path")
    parser.add_argument("--gfx-only", action="store_true",
                        help="Run only GFX inference and output tensor summary without a reference backend")
    parser.add_argument("--per-op-all", action="store_true",
                        help="Forward --per-op-all to ov_gfx_compare_runner for full per-op accuracy output")
    parser.add_argument("--compare-arg", action="append", default=[],
                        help="Extra argument forwarded to ov_gfx_compare_runner. Repeat as needed.")
    parser.add_argument("--adb-serial", help="Optional adb device serial for --platform android")
    parser.add_argument("--remote-dir", default="/data/local/tmp/gfx_eval",
                        help="Staging directory for pushed binaries and models on Android or SSH targets")
    parser.add_argument("--ssh-device-file",
                        help="Device description file with address/login/password/work_directory for --platform ssh")
    parser.add_argument("--ssh-host", help="SSH host for --platform ssh")
    parser.add_argument("--ssh-user", help="SSH user for --platform ssh")
    parser.add_argument("--ssh-password", help="SSH password for --platform ssh")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the staging directory after the run")
    parser.add_argument("--report-json", help="Write parsed results to a JSON report")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    log(f"platform={args.platform} model={args.model}")
    if args.gfx_only and args.per_op_all:
        raise RuntimeError("--gfx-only and --per-op-all are mutually exclusive")
    if not args.gfx_only and args.reference_device == "CPU":
        raise RuntimeError("CPU reference device is not supported; use TEMPLATE")

    ctx: ExecContext
    if args.platform == "android":
        ctx = AndroidExecContext(args.remote_dir, args.adb_serial)
    elif args.platform == "ssh":
        device_config: Dict[str, str] = {}
        if args.ssh_device_file:
            device_config = parse_device_file(args.ssh_device_file)
        host = args.ssh_host or device_config.get("address")
        user = args.ssh_user or device_config.get("login")
        password = args.ssh_password or device_config.get("password")
        if not host or not user or not password:
            raise RuntimeError("--platform ssh requires host/user/password directly or via --ssh-device-file")
        remote_dir = args.remote_dir
        if remote_dir == "/data/local/tmp/gfx_eval":
            work_directory = device_config.get("work_directory", "/tmp")
            remote_dir = f"{work_directory.rstrip('/')}/gfx_eval"
        ctx = SshExecContext(host, user, password, remote_dir)
    else:
        ctx = HostExecContext()

    try:
        runtime_files = list(args.runtime_file)
        if args.platform == "android":
            if not any(Path(item).name == "libc++_shared.so" for item in runtime_files):
                append_runtime_file_if_missing(runtime_files, discover_android_ndk_shared_lib())
        if not args.gfx_only and not args.reference_plugin:
            args.reference_plugin = str(discover_reference_plugin(args.plugin_lib, args.reference_device) or "")
            if args.reference_plugin:
                log(f"auto reference plugin: {args.reference_plugin}")
        if args.reference_plugin and not args.gfx_only:
            append_runtime_file_if_missing(runtime_files, resolve_existing_path(args.reference_plugin))
        model_suffix = Path(args.model).suffix.lower()
        if model_suffix == ".xml":
            append_runtime_file_if_missing(runtime_files,
                                           discover_sibling_artifact(args.plugin_lib, "libopenvino_ir_frontend.so"))

        if runtime_files:
            log(f"staging {len(runtime_files)} runtime files")
        prepared_runtime_files = prepare_runtime_files(ctx, runtime_files)
        plugin_lib = ctx.prepare_runtime_path(args.plugin_lib)
        local_plugin_dir = resolve_existing_path(args.plugin_lib)
        lib_dir_inputs = list(args.lib_dir)
        if local_plugin_dir:
            plugin_parent = str(local_plugin_dir.parent)
            if plugin_parent not in lib_dir_inputs:
                lib_dir_inputs.insert(0, plugin_parent)
        lib_dirs = [ctx.prepare_lib_dir(path) for path in lib_dir_inputs]
        runtime_env = make_runtime_env(plugin_lib, lib_dirs, prepared_runtime_files, args.platform)
        log(f"runtime library paths: {runtime_env}")

        report: Dict[str, object] = {
            "platform": args.platform,
            "model": str(Path(args.model).resolve()) if resolve_existing_path(args.model) else args.model,
        }

        compare = run_compare(ctx, args, runtime_env)
        report["compare"] = {k: v for k, v in compare.items() if k not in ("raw_output",)}
        if compare.get("mode") == "gfx_only":
            for output in compare["outputs"]:
                print(
                    f"GFX_ONLY {output['name']} elements={output['elements']} finite={output['finite']} "
                    f"nan={output['nan']} inf={output['inf']} min={output['min']} max={output['max']} "
                    f"mean={output['mean']} l2={output['l2']}"
                )
        elif compare.get("mode") == "per_op":
            print(f"PER_OP_SUMMARY max_abs={compare['max_abs_diff']} max_rel={compare['max_rel_diff']}")
            print(f"PER_OP count={len(compare['per_op'])}")
        else:
            reference_device = compare.get("reference_device", args.reference_device)
            print(f"ACCURACY {reference_device}_vs_GFX max_abs={compare['max_abs_diff']} max_rel={compare['max_rel_diff']}")
            if compare.get("per_op"):
                print(f"PER_OP count={len(compare['per_op'])}")

        if args.report_json:
            report_path = Path(args.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log(f"wrote report: {report_path}")

        return 0
    finally:
        if args.platform == "host" or args.cleanup:
            ctx.cleanup()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
