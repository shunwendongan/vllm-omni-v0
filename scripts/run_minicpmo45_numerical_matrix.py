#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run restart-isolated MiniCPM-o 4.5 numerical A/B evidence arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CFM_STEPS_ENV = "VLLM_OMNI_MINICPMO45_CFM_STEPS"
FLOW_FP16_ENV = "VLLM_OMNI_MINICPMO45_FLOW_FP16"
OFFICIAL_CONFIG = Path("tests/dfx/perf/tests/test_minicpmo_4_5.json")
PAIR_ORDER = ("B", "C", "C", "B", "B", "C")
WARMUPS = 2
METRIC_NAMES = (
    "mean_audio_rtf",
    "mean_audio_ttfp_ms",
    "mean_ttft_ms",
    "request_throughput",
    "completed",
    "failed",
)


@dataclass(frozen=True)
class Arm:
    name: str
    steps: int
    flow_fp16: bool


ARMS = {
    "B10": Arm("B10", 10, False),
    "S8": Arm("S8", 8, False),
    "S6": Arm("S6", 6, False),
    "H10": Arm("H10", 10, True),
    "H8": Arm("H8", 8, True),
    "H6": Arm("H6", 6, True),
}


@dataclass(frozen=True)
class RunSpec:
    ordinal: int
    pair_index: int
    position: int
    role: str
    arm: Arm


def build_schedule(candidates: list[str]) -> list[RunSpec]:
    if not candidates:
        raise ValueError("at least one numerical candidate is required")
    if len(candidates) != len(set(candidates)):
        raise ValueError("numerical candidates must be unique")
    schedule: list[RunSpec] = []
    ordinal = 0
    for pair_index, candidate_name in enumerate(candidates):
        if candidate_name == "B10":
            raise ValueError("B10 is the fixed baseline and cannot be a candidate")
        try:
            candidate = ARMS[candidate_name]
        except KeyError as exc:
            raise ValueError(f"unknown numerical arm: {candidate_name}") from exc
        for position, role in enumerate(PAIR_ORDER):
            arm = ARMS["B10"] if role == "B" else candidate
            schedule.append(
                RunSpec(
                    ordinal=ordinal,
                    pair_index=pair_index,
                    position=position,
                    role=role,
                    arm=arm,
                )
            )
            ordinal += 1
    return schedule


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tests = payload if isinstance(payload, list) else []
    challenge = next((item for item in tests if item.get("test_name") == "test_minicpmo_4_5_challenge"), None)
    if challenge is None:
        raise ValueError("official config is missing test_minicpmo_4_5_challenge")
    benchmark = challenge["benchmark_params"][0]
    if benchmark.get("seed_tts_locale") != "zh":
        raise ValueError("official MiniCPM-o challenge workload must use Chinese Seed-TTS")
    if benchmark.get("disable_shuffle") is not True:
        raise ValueError("official MiniCPM-o challenge workload must keep request shuffle disabled")
    concurrency = [int(value) for value in benchmark.get("max_concurrency", [])]
    if concurrency != [1, 4, 8]:
        raise ValueError(f"official MiniCPM-o challenge concurrency changed: {concurrency}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "dataset": benchmark.get("dataset_name"),
        "dataset_path": benchmark.get("dataset_path"),
        "locale": benchmark.get("seed_tts_locale"),
        "disable_shuffle": benchmark.get("disable_shuffle"),
        "max_concurrency": concurrency,
        "num_prompts": benchmark.get("num_prompts"),
    }


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _render(command: list[str], values: dict[str, object]) -> list[str]:
    return [str(part).format_map(values) for part in command]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _load_command_config(path: Path | None, *, dry_run: bool) -> dict[str, list[str]]:
    if path is None:
        if dry_run:
            return {}
        raise ValueError("--config is required unless --dry-run is used")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("command config must be a JSON object")
    result: dict[str, list[str]] = {}
    for name in ("server_command", "ready_command", "benchmark_command", "stop_command"):
        value = raw.get(name)
        if value is None and name != "benchmark_command":
            continue
        if not isinstance(value, list) or not value or not all(isinstance(part, str) and part for part in value):
            raise ValueError(f"{name} must be a non-empty JSON string array")
        result[name] = value
    if ("server_command" in result) != ("ready_command" in result):
        raise ValueError("server_command and ready_command must be provided together")
    if "stop_command" in result and "server_command" not in result:
        raise ValueError("stop_command requires server_command and ready_command")
    return result


def create_evidence_root(root: Path, run_id: str | None) -> Path:
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-minicpmo45-numerical")
    destination = root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _arm_environment(base: dict[str, str], arm: Arm) -> dict[str, str]:
    env = dict(base)
    env[CFM_STEPS_ENV] = str(arm.steps)
    env[FLOW_FP16_ENV] = "1" if arm.flow_fp16 else "0"
    return env


def _terminate_server(server: subprocess.Popen[str], timeout: float) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=timeout)


def _wait_ready(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    last_returncode: int | None = None
    while time.monotonic() < deadline:
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        last_returncode = result.returncode
        if result.returncode == 0:
            return
        time.sleep(1.0)
    raise TimeoutError(f"server readiness timed out after {timeout}s (last return code {last_returncode})")


def summarize_json_results(run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, list[float]] = {}
    results: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*.json")):
        if path.name in {"run.json", "summary.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        selected: dict[str, float] = {}
        for name in METRIC_NAMES:
            value = payload.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                selected[name] = float(value)
                metrics.setdefault(name, []).append(float(value))
        completed = selected.get("completed")
        failed = selected.get("failed")
        if completed is not None and failed is not None and completed + failed > 0:
            selected["failure_rate"] = failed / (completed + failed)
            metrics.setdefault("failure_rate", []).append(selected["failure_rate"])
        results.append(
            {
                "path": str(path.relative_to(run_dir)),
                "dataset_name": payload.get("dataset_name"),
                "max_concurrency": payload.get("max_concurrency"),
                "num_prompts": payload.get("num_prompts"),
                "metrics": selected,
            }
        )
    return {"json_files": [result["path"] for result in results], "results": results, "metrics": metrics}


def _run_one(
    spec: RunSpec,
    *,
    evidence_root: Path,
    repo: Path,
    official_config: Path,
    command_config: dict[str, list[str]],
    ready_timeout: float,
    stop_timeout: float,
    dry_run: bool,
) -> dict[str, Any]:
    run_dir = evidence_root / f"{spec.ordinal:02d}-p{spec.pair_index}-{spec.position}-{spec.role}-{spec.arm.name}"
    run_dir.mkdir(exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "benchmark" / "raw").mkdir(parents=True)
    values: dict[str, object] = {
        "arm": spec.arm.name,
        "steps": spec.arm.steps,
        "flow_fp16": int(spec.arm.flow_fp16),
        "warmups": WARMUPS,
        "run_dir": str(run_dir.resolve()),
        "benchmark_dir": str((run_dir / "benchmark" / "raw").resolve()),
        "official_config": str(official_config.resolve()),
    }
    commands = {name: _render(command, values) for name, command in command_config.items()}
    run_record = {
        "spec": {**asdict(spec), "arm": asdict(spec.arm)},
        "environment": {
            CFM_STEPS_ENV: str(spec.arm.steps),
            FLOW_FP16_ENV: "1" if spec.arm.flow_fp16 else "0",
        },
        "commands": commands,
        "server_mode": "explicit" if "server_command" in commands else "benchmark-managed",
        "warmups": WARMUPS,
        "status": "planned" if dry_run else "running",
    }
    (run_dir / "run.json").write_text(json.dumps(run_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        summary = {**run_record, "status": "dry-run"}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    env = _arm_environment(os.environ, spec.arm)
    env["BENCHMARK_DIR"] = str((run_dir / "benchmark" / "raw").resolve())
    command_log = evidence_root / "commands.jsonl"
    server_log_path = run_dir / "logs" / "service.log"
    benchmark_log_path = run_dir / "logs" / "benchmark.log"
    server: subprocess.Popen[str] | None = None
    summary: dict[str, Any]
    try:
        if "server_command" in commands:
            with server_log_path.open("w", encoding="utf-8") as server_log:
                _append_jsonl(
                    command_log,
                    {"event": "server_start", "run": spec.ordinal, "argv": commands["server_command"]},
                )
                server = subprocess.Popen(
                    commands["server_command"],
                    cwd=repo,
                    env=env,
                    text=True,
                    stdout=server_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            _wait_ready(
                commands["ready_command"],
                cwd=repo,
                env=env,
                timeout=ready_timeout,
                log_path=run_dir / "logs" / "readiness.log",
            )
        started = time.time()
        with benchmark_log_path.open("w", encoding="utf-8") as benchmark_log:
            result = subprocess.run(
                commands["benchmark_command"],
                cwd=repo,
                env=env,
                text=True,
                stdout=benchmark_log,
                stderr=subprocess.STDOUT,
            )
        _append_jsonl(
            command_log,
            {
                "event": "benchmark",
                "run": spec.ordinal,
                "argv": commands["benchmark_command"],
                "returncode": result.returncode,
                "elapsed_s": time.time() - started,
            },
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, commands["benchmark_command"])
        summary = {**run_record, **summarize_json_results(run_dir), "status": "completed"}
    except BaseException as exc:
        summary = {
            **run_record,
            **summarize_json_results(run_dir),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        try:
            if server is not None:
                if "stop_command" in commands:
                    subprocess.run(commands["stop_command"], cwd=repo, env=env, check=False)
                _terminate_server(server, stop_timeout)
        except BaseException as exc:
            summary["status"] = "failed"
            summary["cleanup_error_type"] = type(exc).__name__
            summary["cleanup_error"] = str(exc)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def summarize_matrix_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for run in runs:
        arm_name = str(run["spec"]["arm"]["name"])
        arm = by_arm.setdefault(arm_name, {"completed_runs": 0, "failed_runs": 0, "metrics": {}})
        status_key = "completed_runs" if run.get("status") == "completed" else "failed_runs"
        arm[status_key] += 1
        if run.get("status") != "completed":
            continue
        for name, values in run.get("metrics", {}).items():
            arm["metrics"].setdefault(name, []).extend(values)
    return by_arm


def _host_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "npu_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "torch_npu_allocator_config": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON argv templates for server/readiness/benchmark")
    parser.add_argument(
        "--candidates",
        default="S8,S6,H10,H8,H6",
        help="comma-separated candidate arms; B10 is inserted as the baseline",
    )
    parser.add_argument("--root", type=Path, default=Path("work/perf-evidence"))
    parser.add_argument("--run-id", help="unique evidence directory name; existing directories are rejected")
    parser.add_argument("--official-config", type=Path, default=OFFICIAL_CONFIG)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--stop-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    candidates = [value.strip().upper() for value in args.candidates.split(",") if value.strip()]
    schedule = build_schedule(candidates)
    command_config = _load_command_config(args.config, dry_run=args.dry_run)
    official_path = args.official_config if args.official_config.is_absolute() else repo / args.official_config
    official = validate_official_config(official_path)
    evidence_root = create_evidence_root(args.root if args.root.is_absolute() else repo / args.root, args.run_id)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry-run" if args.dry_run else "running",
        "repository": str(repo),
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "host_environment": _host_environment(),
        "official_config": official,
        "pair_order": list(PAIR_ORDER),
        "warmups": WARMUPS,
        "candidates": candidates,
        "schedule": [{**asdict(spec), "arm": asdict(spec.arm)} for spec in schedule],
    }
    (evidence_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summaries: list[dict[str, Any]] = []
    try:
        for spec in schedule:
            summaries.append(
                _run_one(
                    spec,
                    evidence_root=evidence_root,
                    repo=repo,
                    official_config=official_path,
                    command_config=command_config,
                    ready_timeout=args.ready_timeout,
                    stop_timeout=args.stop_timeout,
                    dry_run=args.dry_run,
                )
            )
            if summaries[-1]["status"] == "failed":
                break
        manifest["status"] = (
            "failed"
            if summaries and summaries[-1]["status"] == "failed"
            else ("dry-run" if args.dry_run else "completed")
        )
        returncode = 1 if manifest["status"] == "failed" else 0
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["driver_error_type"] = type(exc).__name__
        manifest["driver_error"] = str(exc)
        returncode = 1
    finally:
        manifest["runs"] = summaries
        manifest["by_arm"] = summarize_matrix_runs(summaries)
        (evidence_root / "matrix_summary.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(evidence_root)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
