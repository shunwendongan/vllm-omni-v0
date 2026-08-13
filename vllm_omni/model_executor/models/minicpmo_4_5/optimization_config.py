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
INITIAL_STATE_CACHE_ENV = "VLLM_OMNI_MINICPMO45_INITIAL_STATE_CACHE"
INITIAL_STATE_CACHE_MAX_ENTRIES_ENV = "VLLM_OMNI_MINICPMO45_INITIAL_STATE_CACHE_MAX_ENTRIES"
BATCH1_LOW_COPY_ENV = "VLLM_OMNI_MINICPMO45_BATCH1_LOW_COPY"
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


def _parse_non_negative_int(
    env: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    raw_value = env.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name}={raw_value!r}; expected a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"Invalid {name}={raw_value!r}; expected a non-negative integer")
    return value


@dataclass(frozen=True)
class MiniCPMO45OptimizationConfig:
    """Process-local switches, parsed once so invalid values fail at startup."""

    tensor_handoff: bool = False
    initial_state_cache: bool = False
    initial_state_cache_max_entries: int = 1
    batch1_low_copy: bool = False
    perf_stats: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> MiniCPMO45OptimizationConfig:
        values = os.environ if env is None else env
        config = cls(
            tensor_handoff=_parse_bool(values, TENSOR_HANDOFF_ENV),
            initial_state_cache=_parse_bool(values, INITIAL_STATE_CACHE_ENV),
            initial_state_cache_max_entries=_parse_non_negative_int(
                values,
                INITIAL_STATE_CACHE_MAX_ENTRIES_ENV,
                default=1,
            ),
            batch1_low_copy=_parse_bool(values, BATCH1_LOW_COPY_ENV),
            perf_stats=_parse_bool(values, PERF_STATS_ENV),
        )
        if config.initial_state_cache and config.initial_state_cache_max_entries < 1:
            raise ValueError(
                f"Invalid {INITIAL_STATE_CACHE_MAX_ENTRIES_ENV}="
                f"{config.initial_state_cache_max_entries!r}; expected >= 1 when "
                f"{INITIAL_STATE_CACHE_ENV}=1"
            )
        return config


@dataclass
class MiniCPMO45PerfStats:
    """Host-only counters; recording never synchronizes an accelerator."""

    tensor_handoff_count: int = 0
    legacy_handoff_count: int = 0
    handoff_prepare_ns: int = 0
    initial_state_cache_hit_count: int = 0
    initial_state_cache_miss_count: int = 0
    initial_state_cache_evict_count: int = 0
    initial_state_setup_ns: int = 0
    initial_state_clone_ns: int = 0
    batch1_low_copy_count: int = 0
    batch1_flow_stack_cat_skipped_count: int = 0
    batch1_flow_split_cat_skipped_count: int = 0
    batch1_flow_clone_skipped_count: int = 0
    batch1_hift_stack_cat_skipped_count: int = 0
    batch1_hift_clone_skipped_count: int = 0

    def record_handoff(self, *, tensor_path: bool, elapsed_ns: int) -> None:
        if tensor_path:
            self.tensor_handoff_count += 1
        else:
            self.legacy_handoff_count += 1
        self.handoff_prepare_ns += elapsed_ns

    def record_initial_state_cache(self, *, hit: bool) -> None:
        if hit:
            self.initial_state_cache_hit_count += 1
        else:
            self.initial_state_cache_miss_count += 1

    def record_initial_state_cache_evict(self) -> None:
        self.initial_state_cache_evict_count += 1

    def record_initial_state_setup(self, *, elapsed_ns: int) -> None:
        self.initial_state_setup_ns += elapsed_ns

    def record_initial_state_clone(self, *, elapsed_ns: int) -> None:
        self.initial_state_clone_ns += elapsed_ns

    def record_batch1_low_copy(
        self,
        *,
        flow_stack_cats: int,
        flow_split_cats: int,
        flow_clones: int,
        hift_stack_cats: int,
        hift_clones: int,
    ) -> None:
        self.batch1_low_copy_count += 1
        self.batch1_flow_stack_cat_skipped_count += flow_stack_cats
        self.batch1_flow_split_cat_skipped_count += flow_split_cats
        self.batch1_flow_clone_skipped_count += flow_clones
        self.batch1_hift_stack_cat_skipped_count += hift_stack_cats
        self.batch1_hift_clone_skipped_count += hift_clones

    def snapshot(self) -> dict[str, int]:
        return {
            "tensor_handoff_count": self.tensor_handoff_count,
            "legacy_handoff_count": self.legacy_handoff_count,
            "handoff_prepare_ns": self.handoff_prepare_ns,
            "initial_state_cache_hit_count": self.initial_state_cache_hit_count,
            "initial_state_cache_miss_count": self.initial_state_cache_miss_count,
            "initial_state_cache_evict_count": self.initial_state_cache_evict_count,
            "initial_state_setup_ns": self.initial_state_setup_ns,
            "initial_state_clone_ns": self.initial_state_clone_ns,
            "batch1_low_copy_count": self.batch1_low_copy_count,
            "batch1_flow_stack_cat_skipped_count": self.batch1_flow_stack_cat_skipped_count,
            "batch1_flow_split_cat_skipped_count": self.batch1_flow_split_cat_skipped_count,
            "batch1_flow_clone_skipped_count": self.batch1_flow_clone_skipped_count,
            "batch1_hift_stack_cat_skipped_count": self.batch1_hift_stack_cat_skipped_count,
            "batch1_hift_clone_skipped_count": self.batch1_hift_clone_skipped_count,
        }

    def reset(self) -> None:
        self.tensor_handoff_count = 0
        self.legacy_handoff_count = 0
        self.handoff_prepare_ns = 0
        self.initial_state_cache_hit_count = 0
        self.initial_state_cache_miss_count = 0
        self.initial_state_cache_evict_count = 0
        self.initial_state_setup_ns = 0
        self.initial_state_clone_ns = 0
        self.batch1_low_copy_count = 0
        self.batch1_flow_stack_cat_skipped_count = 0
        self.batch1_flow_split_cat_skipped_count = 0
        self.batch1_flow_clone_skipped_count = 0
        self.batch1_hift_stack_cat_skipped_count = 0
        self.batch1_hift_clone_skipped_count = 0


MINICPMO45_OPTIMIZATION_CONFIG = MiniCPMO45OptimizationConfig.from_env()
MINICPMO45_PERF_STATS = MiniCPMO45PerfStats()
