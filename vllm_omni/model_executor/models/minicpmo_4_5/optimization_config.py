# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MiniCPM-o 4.5 optimization switches and host counters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

TENSOR_HANDOFF_ENV = "VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF"
PERF_STATS_ENV = "VLLM_OMNI_MINICPMO45_PERF_STATS"


def _parse_bool(env: Mapping[str, str], name: str, *, default: bool = False) -> bool:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    choices = sorted(_TRUE_VALUES | _FALSE_VALUES)
    raise ValueError(f"Invalid {name}={raw_value!r}; expected one of {choices}")


@dataclass(frozen=True)
class MiniCPMO45OptimizationConfig:
    """Process-local switches, parsed once so invalid values fail at startup."""

    tensor_handoff: bool = False
    perf_stats: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MiniCPMO45OptimizationConfig:
        values = os.environ if env is None else env
        return cls(
            tensor_handoff=_parse_bool(values, TENSOR_HANDOFF_ENV),
            perf_stats=_parse_bool(values, PERF_STATS_ENV),
        )


@dataclass
class MiniCPMO45PerfStats:
    """Host-only counters; recording never synchronizes an accelerator."""

    tensor_handoff_count: int = 0
    legacy_handoff_count: int = 0
    handoff_prepare_ns: int = 0

    def record_handoff(self, *, tensor_path: bool, elapsed_ns: int) -> None:
        if tensor_path:
            self.tensor_handoff_count += 1
        else:
            self.legacy_handoff_count += 1
        self.handoff_prepare_ns += elapsed_ns

    def snapshot(self) -> dict[str, int]:
        return {
            "tensor_handoff_count": self.tensor_handoff_count,
            "legacy_handoff_count": self.legacy_handoff_count,
            "handoff_prepare_ns": self.handoff_prepare_ns,
        }

    def reset(self) -> None:
        self.tensor_handoff_count = 0
        self.legacy_handoff_count = 0
        self.handoff_prepare_ns = 0


MINICPMO45_OPTIMIZATION_CONFIG = MiniCPMO45OptimizationConfig.from_env()
MINICPMO45_PERF_STATS = MiniCPMO45PerfStats()
