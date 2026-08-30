# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark MiniCPM-o tensor raw-bytes vs legacy nested-list handoff.

This is a Host-only microbenchmark. It does not establish end-to-end or Atlas
A3 performance; use it to isolate representation/materialization overhead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import msgspec
import torch

from vllm_omni.engine.serialization import (
    deserialize_model_intermediate_buffer,
    serialize_model_intermediate_buffer,
)


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _summary(values: list[int]) -> dict[str, float | int]:
    return {
        "iterations": len(values),
        "mean_ns": statistics.fmean(values),
        "median_ns": statistics.median(values),
        "p90_ns": _percentile(values, 0.90),
        "p95_ns": _percentile(values, 0.95),
        "p99_ns": _percentile(values, 0.99),
    }


def _round_trip(source: torch.Tensor, *, raw: bool) -> tuple[int, torch.Tensor]:
    payload_value: Any = source if raw else source.tolist()
    prepared = serialize_model_intermediate_buffer(
        {
            "request_id": "tensor-handoff-benchmark",
            "hidden_states": {"tts": payload_value},
        }
    )
    encoded = msgspec.msgpack.encode(prepared)
    decoded = msgspec.msgpack.decode(encoded)
    restored = deserialize_model_intermediate_buffer(decoded)
    result = restored["hidden_states"]["tts"]
    if not isinstance(result, torch.Tensor):
        result = torch.tensor(result, dtype=source.dtype)
    return len(encoded), result


def run_case(
    source: torch.Tensor,
    *,
    raw: bool,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _round_trip(source, raw=raw)

    timings: list[int] = []
    payload_bytes = 0
    restored: torch.Tensor | None = None
    for _ in range(repeat):
        started = time.perf_counter_ns()
        payload_bytes, restored = _round_trip(source, raw=raw)
        timings.append(time.perf_counter_ns() - started)

    assert restored is not None
    if not torch.equal(restored, source):
        raise RuntimeError("tensor handoff round-trip changed values")
    return {
        "path": "raw" if raw else "legacy",
        "payload_bytes": payload_bytes,
        **_summary(timings),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.rows < 0 or args.cols < 0 or args.warmup < 0 or args.repeat < 1:
        parser.error("rows/cols/warmup must be non-negative and repeat must be >= 1")

    generator = torch.Generator().manual_seed(args.seed)
    source = torch.randn((args.rows, args.cols), generator=generator, dtype=torch.float32)
    results = [
        run_case(source, raw=True, warmup=args.warmup, repeat=args.repeat),
        run_case(source, raw=False, warmup=args.warmup, repeat=args.repeat),
    ]
    payload = {
        "benchmark": "minicpmo45_tensor_handoff_host_only",
        "shape": list(source.shape),
        "dtype": str(source.dtype),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
