# SPDX-License-Identifier: Apache-2.0
"""Scoped MiniCPM-o 4.5 NPU fast-path configuration.

The performance path spans config resolution, the scheduler, the NPU runner,
and Code2Wav.  Keeping its knobs in one immutable object prevents the runner
and scheduler from silently using different decode window sizes.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _parse_bool_env(name: str, default: bool) -> tuple[bool, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True, True
    if normalized in _FALSE_VALUES:
        return False, True
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
    )


def _parse_int_env(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, bool]:
    raw = os.environ.get(name)
    if raw is None:
        return default, False
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be in [{minimum}, {maximum}], got {value}"
        )
    return value, True


@dataclass(frozen=True)
class MiniCPMO45FastPathConfig:
    """Resolved settings shared by every MiniCPM-o 4.5 stage."""

    enabled: bool
    scope_active: bool
    talker_local_steps: int
    ngram_spec_tokens: int
    prompt_state_cache: bool
    prewarm: bool
    prewarm_ready_gate: bool
    prompt_cache_max_entries: int
    prompt_cache_max_bytes: int
    talker_steps_explicit: bool = False
    ngram_tokens_explicit: bool = False
    prompt_state_cache_explicit: bool = False
    prewarm_explicit: bool = False
    prewarm_ready_gate_explicit: bool = False
    prompt_cache_max_entries_explicit: bool = False
    prompt_cache_max_bytes_explicit: bool = False

    @property
    def talker_enabled(self) -> bool:
        return self.enabled and self.scope_active and self.talker_local_steps > 1

    @property
    def ngram_enabled(self) -> bool:
        return self.enabled and self.scope_active and self.ngram_spec_tokens > 0

    def for_stage(self, stage_id: int) -> "MiniCPMO45FastPathConfig":
        """Return a stage-scoped view without changing the wire schema."""
        if stage_id == 0:
            return replace(
                self,
                talker_local_steps=1,
                prompt_state_cache=False,
                prewarm=False,
                prewarm_ready_gate=False,
            )
        if stage_id == 1:
            return replace(
                self,
                ngram_spec_tokens=0,
                prompt_state_cache=False,
                prewarm=False,
                prewarm_ready_gate=False,
            )
        if stage_id == 2:
            return replace(
                self,
                talker_local_steps=1,
                ngram_spec_tokens=0,
            )
        return replace(
            self,
            enabled=False,
            talker_local_steps=1,
            ngram_spec_tokens=0,
            prompt_state_cache=False,
            prewarm=False,
            prewarm_ready_gate=False,
        )

    def to_stage_dict(self, stage_id: int) -> dict[str, Any]:
        return asdict(self.for_stage(stage_id))

    def with_deploy_overrides(
        self,
        value: Mapping[str, Any] | None,
    ) -> "MiniCPMO45FastPathConfig":
        """Apply deploy values below explicit environment variables.

        The master switch and the model/device capability gate have already
        been resolved before this method is called.  Consequently a disabled
        config ignores deploy feature values entirely, which makes
        ``VLLM_OMNI_MINICPMO45_FASTPATH=0`` an unconditional rollback.
        """
        if not self.enabled or value is None:
            return self
        if not isinstance(value, Mapping):
            raise TypeError("additional_config.minicpmo45_fastpath must be a mapping")

        allowed = {
            "talker_local_steps",
            "ngram_spec_tokens",
            "prompt_state_cache",
            "prewarm",
            "prewarm_ready_gate",
            "prompt_cache_max_entries",
            "prompt_cache_max_bytes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Unknown deploy MiniCPM-o 4.5 fast-path settings: "
                + ", ".join(sorted(unknown))
            )

        updates: dict[str, Any] = {}
        if "talker_local_steps" in value and not self.talker_steps_explicit:
            updates["talker_local_steps"] = _coerce_int(
                "additional_config.minicpmo45_fastpath.talker_local_steps",
                value["talker_local_steps"],
                minimum=1,
                maximum=32,
            )
            updates["talker_steps_explicit"] = True
        if "ngram_spec_tokens" in value and not self.ngram_tokens_explicit:
            updates["ngram_spec_tokens"] = _coerce_int(
                "additional_config.minicpmo45_fastpath.ngram_spec_tokens",
                value["ngram_spec_tokens"],
                minimum=0,
                maximum=32,
            )
            updates["ngram_tokens_explicit"] = True
        if "prompt_state_cache" in value and not self.prompt_state_cache_explicit:
            updates["prompt_state_cache"] = _coerce_bool(
                "additional_config.minicpmo45_fastpath.prompt_state_cache",
                value["prompt_state_cache"],
            )
            updates["prompt_state_cache_explicit"] = True
        if "prewarm" in value and not self.prewarm_explicit:
            updates["prewarm"] = _coerce_bool(
                "additional_config.minicpmo45_fastpath.prewarm",
                value["prewarm"],
            )
            updates["prewarm_explicit"] = True
        if (
            "prewarm_ready_gate" in value
            and not self.prewarm_ready_gate_explicit
        ):
            updates["prewarm_ready_gate"] = _coerce_bool(
                "additional_config.minicpmo45_fastpath.prewarm_ready_gate",
                value["prewarm_ready_gate"],
            )
            updates["prewarm_ready_gate_explicit"] = True
        if (
            "prompt_cache_max_entries" in value
            and not self.prompt_cache_max_entries_explicit
        ):
            updates["prompt_cache_max_entries"] = _coerce_int(
                "additional_config.minicpmo45_fastpath.prompt_cache_max_entries",
                value["prompt_cache_max_entries"],
                minimum=1,
                maximum=4096,
            )
            updates["prompt_cache_max_entries_explicit"] = True
        if (
            "prompt_cache_max_bytes" in value
            and not self.prompt_cache_max_bytes_explicit
        ):
            updates["prompt_cache_max_bytes"] = _coerce_int(
                "additional_config.minicpmo45_fastpath.prompt_cache_max_bytes",
                value["prompt_cache_max_bytes"],
                minimum=1,
                maximum=1 << 50,
            )
            updates["prompt_cache_max_bytes_explicit"] = True

        resolved = replace(self, **updates)
        if not resolved.prewarm:
            resolved = replace(resolved, prewarm_ready_gate=False)
        return resolved

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "MiniCPMO45FastPathConfig":
        """Rebuild a config embedded in ``VllmConfig.additional_config``."""
        if value is None:
            return disabled_minicpmo45_fastpath()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                "Unknown MiniCPM-o 4.5 fast-path settings: "
                + ", ".join(sorted(unknown))
            )
        return cls(**dict(value))


def disabled_minicpmo45_fastpath() -> MiniCPMO45FastPathConfig:
    return MiniCPMO45FastPathConfig(
        enabled=False,
        scope_active=False,
        talker_local_steps=1,
        ngram_spec_tokens=0,
        prompt_state_cache=False,
        prewarm=False,
        prewarm_ready_gate=False,
        prompt_cache_max_entries=16,
        prompt_cache_max_bytes=512 * 1024 * 1024,
    )


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(
        f"{name} must be a boolean or one of "
        f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {value!r}"
    )


def _coerce_int(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if not minimum <= parsed <= maximum:
        raise ValueError(
            f"{name} must be in [{minimum}, {maximum}], got {parsed}"
        )
    return parsed


def resolve_minicpmo45_fastpath(
    *,
    model_type: str,
    is_npu: bool,
) -> MiniCPMO45FastPathConfig:
    """Parse the process environment and apply the NPU/model capability gate.

    Environment values are validated even outside the active scope.  A typo
    must fail at startup rather than remain latent until a deployment moves to
    NPU.
    """

    enabled, _ = _parse_bool_env("VLLM_OMNI_MINICPMO45_FASTPATH", True)
    talker_steps, talker_explicit = _parse_int_env(
        "VLLM_OMNI_MINICPMO45_TALKER_LOCAL_STEPS",
        12,
        minimum=1,
        maximum=32,
    )
    ngram_tokens, ngram_explicit = _parse_int_env(
        "VLLM_OMNI_MINICPMO45_NGRAM_SPEC_TOKENS",
        14,
        minimum=0,
        maximum=32,
    )
    prompt_state_cache, prompt_state_cache_explicit = _parse_bool_env(
        "VLLM_OMNI_MINICPMO45_PROMPT_STATE_CACHE",
        True,
    )
    prewarm, prewarm_explicit = _parse_bool_env(
        "VLLM_OMNI_MINICPMO45_PREWARM", True
    )
    prewarm_ready_gate, prewarm_ready_gate_explicit = _parse_bool_env(
        "VLLM_OMNI_MINICPMO45_PREWARM_READY_GATE",
        True,
    )
    max_entries, max_entries_explicit = _parse_int_env(
        "VLLM_OMNI_MINICPMO45_PROMPT_CACHE_MAX_ENTRIES",
        16,
        minimum=1,
        maximum=4096,
    )
    max_bytes, max_bytes_explicit = _parse_int_env(
        "VLLM_OMNI_MINICPMO45_PROMPT_CACHE_MAX_BYTES",
        512 * 1024 * 1024,
        minimum=1,
        maximum=1 << 50,
    )

    scope_active = model_type == "minicpmo_4_5" and is_npu
    active = enabled and scope_active
    return MiniCPMO45FastPathConfig(
        enabled=active,
        scope_active=scope_active,
        talker_local_steps=talker_steps if active else 1,
        ngram_spec_tokens=ngram_tokens if active else 0,
        prompt_state_cache=prompt_state_cache and active,
        prewarm=prewarm and active,
        prewarm_ready_gate=prewarm_ready_gate and prewarm and active,
        prompt_cache_max_entries=max_entries,
        prompt_cache_max_bytes=max_bytes,
        talker_steps_explicit=talker_explicit,
        ngram_tokens_explicit=ngram_explicit,
        prompt_state_cache_explicit=prompt_state_cache_explicit,
        prewarm_explicit=prewarm_explicit,
        prewarm_ready_gate_explicit=prewarm_ready_gate_explicit,
        prompt_cache_max_entries_explicit=max_entries_explicit,
        prompt_cache_max_bytes_explicit=max_bytes_explicit,
    )


def fastpath_from_vllm_config(vllm_config: Any) -> MiniCPMO45FastPathConfig:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, Mapping):
        return disabled_minicpmo45_fastpath()
    raw = additional.get("minicpmo45_fastpath")
    if raw is None:
        return disabled_minicpmo45_fastpath()
    if not isinstance(raw, Mapping):
        raise TypeError("additional_config.minicpmo45_fastpath must be a mapping")
    return MiniCPMO45FastPathConfig.from_mapping(raw)
