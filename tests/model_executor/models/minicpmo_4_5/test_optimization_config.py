# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration tests for opt-in MiniCPM-o 4.5 optimizations."""

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.optimization_config import (
    PERF_STATS_ENV,
    TENSOR_HANDOFF_ENV,
    MiniCPMO45OptimizationConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_optimization_flags_default_to_disabled() -> None:
    config = MiniCPMO45OptimizationConfig.from_env({})

    assert config.tensor_handoff is False
    assert config.perf_stats is False


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_tensor_handoff_accepts_explicit_true_values(value: str) -> None:
    config = MiniCPMO45OptimizationConfig.from_env({TENSOR_HANDOFF_ENV: value})

    assert config.tensor_handoff is True


def test_perf_stats_can_be_enabled_independently() -> None:
    config = MiniCPMO45OptimizationConfig.from_env({PERF_STATS_ENV: "1"})

    assert config.tensor_handoff is False
    assert config.perf_stats is True


def test_invalid_boolean_fails_fast() -> None:
    with pytest.raises(ValueError, match=TENSOR_HANDOFF_ENV):
        MiniCPMO45OptimizationConfig.from_env({TENSOR_HANDOFF_ENV: "sometimes"})
