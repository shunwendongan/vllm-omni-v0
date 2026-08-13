# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration tests for opt-in MiniCPM-o 4.5 optimizations."""

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.optimization_config import (
    BATCH1_LOW_COPY_ENV,
    INITIAL_STATE_CACHE_ENV,
    INITIAL_STATE_CACHE_MAX_ENTRIES_ENV,
    PERF_STATS_ENV,
    TENSOR_HANDOFF_ENV,
    MiniCPMO45OptimizationConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_optimization_flags_default_to_disabled() -> None:
    config = MiniCPMO45OptimizationConfig.from_env({})

    assert config.tensor_handoff is False
    assert config.initial_state_cache is False
    assert config.initial_state_cache_max_entries == 1
    assert config.batch1_low_copy is False
    assert config.perf_stats is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_tensor_handoff_accepts_explicit_true_values(value: str) -> None:
    config = MiniCPMO45OptimizationConfig.from_env({TENSOR_HANDOFF_ENV: value})

    assert config.tensor_handoff is True


def test_perf_stats_can_be_enabled_independently() -> None:
    config = MiniCPMO45OptimizationConfig.from_env({PERF_STATS_ENV: "1"})

    assert config.tensor_handoff is False
    assert config.perf_stats is True


def test_batch1_low_copy_can_be_enabled_independently() -> None:
    config = MiniCPMO45OptimizationConfig.from_env({BATCH1_LOW_COPY_ENV: "1"})

    assert config.batch1_low_copy is True


def test_all_candidate_flags_can_be_enabled_together() -> None:
    config = MiniCPMO45OptimizationConfig.from_env(
        {
            TENSOR_HANDOFF_ENV: "1",
            INITIAL_STATE_CACHE_ENV: "1",
            INITIAL_STATE_CACHE_MAX_ENTRIES_ENV: "2",
            BATCH1_LOW_COPY_ENV: "1",
            PERF_STATS_ENV: "1",
        }
    )

    assert config == MiniCPMO45OptimizationConfig(
        tensor_handoff=True,
        initial_state_cache=True,
        initial_state_cache_max_entries=2,
        batch1_low_copy=True,
        perf_stats=True,
    )


def test_invalid_batch1_low_copy_boolean_fails_fast() -> None:
    with pytest.raises(ValueError, match=BATCH1_LOW_COPY_ENV):
        MiniCPMO45OptimizationConfig.from_env({BATCH1_LOW_COPY_ENV: "auto"})


def test_invalid_boolean_fails_fast() -> None:
    with pytest.raises(ValueError, match=TENSOR_HANDOFF_ENV):
        MiniCPMO45OptimizationConfig.from_env({TENSOR_HANDOFF_ENV: "sometimes"})


def test_initial_state_cache_parses_capacity() -> None:
    config = MiniCPMO45OptimizationConfig.from_env(
        {
            INITIAL_STATE_CACHE_ENV: "1",
            INITIAL_STATE_CACHE_MAX_ENTRIES_ENV: "3",
        }
    )

    assert config.initial_state_cache is True
    assert config.initial_state_cache_max_entries == 3


@pytest.mark.parametrize("value", ["-1", "1.5", "many", ""])
def test_invalid_initial_state_cache_capacity_fails_fast(value: str) -> None:
    with pytest.raises(ValueError, match=INITIAL_STATE_CACHE_MAX_ENTRIES_ENV):
        MiniCPMO45OptimizationConfig.from_env({INITIAL_STATE_CACHE_MAX_ENTRIES_ENV: value})


def test_zero_capacity_is_only_valid_while_cache_is_disabled() -> None:
    disabled = MiniCPMO45OptimizationConfig.from_env({INITIAL_STATE_CACHE_MAX_ENTRIES_ENV: "0"})
    assert disabled.initial_state_cache_max_entries == 0

    with pytest.raises(ValueError, match=INITIAL_STATE_CACHE_MAX_ENTRIES_ENV):
        MiniCPMO45OptimizationConfig.from_env(
            {
                INITIAL_STATE_CACHE_ENV: "1",
                INITIAL_STATE_CACHE_MAX_ENTRIES_ENV: "0",
            }
        )
