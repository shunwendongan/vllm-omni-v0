"""Shared serialization helpers for omni engine request payloads."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload, deserialize_payload, serialize_payload
from vllm_omni.benchmarks.ultra_timeline import emit_ultra_timeline_event
from vllm_omni.engine import AdditionalInformationPayload

logger = init_logger(__name__)

_MODEL_BUFFER_TENSOR_MARKER = "__vllm_omni_model_buffer_tensor__"
_MODEL_BUFFER_TENSOR_SCHEMA_VERSION = 1
_MODEL_BUFFER_TENSOR_KEYS = frozenset(
    {
        _MODEL_BUFFER_TENSOR_MARKER,
        "dtype",
        "shape",
        "data",
    }
)


@dataclass
class ModelBufferHandoffTelemetry:
    serialize_calls: int = 0
    serialize_ns: int = 0
    serialize_payload_bytes: int = 0
    serialize_raw_path: int = 0
    serialize_legacy_path: int = 0
    deserialize_calls: int = 0
    deserialize_ns: int = 0
    deserialize_payload_bytes: int = 0
    deserialize_raw_path: int = 0
    deserialize_legacy_path: int = 0


@dataclass
class _OperationStats:
    tensor_count: int = 0
    payload_bytes: int = 0


_HANDOFF_TELEMETRY = ModelBufferHandoffTelemetry()
_HANDOFF_TELEMETRY_LOCK = threading.Lock()


def get_model_buffer_handoff_telemetry(
    *,
    reset: bool = False,
) -> dict[str, int]:
    """Read process-local Host counters, optionally resetting atomically."""
    global _HANDOFF_TELEMETRY
    with _HANDOFF_TELEMETRY_LOCK:
        snapshot = asdict(_HANDOFF_TELEMETRY)
        if reset:
            _HANDOFF_TELEMETRY = ModelBufferHandoffTelemetry()
    return snapshot


def reset_model_buffer_handoff_telemetry() -> None:
    get_model_buffer_handoff_telemetry(reset=True)


def _record_handoff(
    operation: str,
    *,
    elapsed_ns: int,
    stats: _OperationStats,
) -> None:
    path = "raw" if stats.tensor_count else "legacy"
    with _HANDOFF_TELEMETRY_LOCK:
        setattr(
            _HANDOFF_TELEMETRY,
            f"{operation}_calls",
            getattr(_HANDOFF_TELEMETRY, f"{operation}_calls") + 1,
        )
        setattr(
            _HANDOFF_TELEMETRY,
            f"{operation}_ns",
            getattr(_HANDOFF_TELEMETRY, f"{operation}_ns") + elapsed_ns,
        )
        setattr(
            _HANDOFF_TELEMETRY,
            f"{operation}_payload_bytes",
            getattr(_HANDOFF_TELEMETRY, f"{operation}_payload_bytes")
            + stats.payload_bytes,
        )
        path_field = f"{operation}_{path}_path"
        setattr(
            _HANDOFF_TELEMETRY,
            path_field,
            getattr(_HANDOFF_TELEMETRY, path_field) + 1,
        )


def _buffer_request_id(buffer: Mapping[str, Any]) -> str | None:
    value = buffer.get("global_request_id") or buffer.get("request_id")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value is not None else None


def _emit_handoff_event(
    operation: str,
    *,
    buffer: Mapping[str, Any],
    elapsed_ns: int,
    stats: _OperationStats,
) -> None:
    emit_ultra_timeline_event(
        f"tensor_handoff_{operation}",
        request_id=_buffer_request_id(buffer),
        stage="engine",
        stream="serialization",
        num_bytes=stats.payload_bytes,
        details={
            "path": "raw" if stats.tensor_count else "legacy",
            "elapsed_ns": elapsed_ns,
            "payload_bytes": stats.payload_bytes,
            "tensor_count": stats.tensor_count,
        },
    )


def _is_model_buffer_tensor_envelope(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == _MODEL_BUFFER_TENSOR_KEYS
        and value.get(_MODEL_BUFFER_TENSOR_MARKER) == _MODEL_BUFFER_TENSOR_SCHEMA_VERSION
    )


def _serialize_model_buffer_value(value: Any, stats: _OperationStats) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.layout != torch.strided:
            raise TypeError(
                "model_intermediate_buffer tensor must use strided layout; "
                f"received layout={tensor.layout}"
            )
        tensor = tensor.to(device="cpu").contiguous()
        try:
            byte_view = tensor.reshape(-1).view(torch.uint8)
            data = byte_view.numpy().tobytes()
        except (RuntimeError, TypeError) as exc:
            raise TypeError(
                "model_intermediate_buffer tensor cannot be serialized: "
                f"dtype={tensor.dtype} shape={tuple(tensor.shape)} "
                f"layout={tensor.layout}"
            ) from exc
        stats.tensor_count += 1
        stats.payload_bytes += len(data)
        return {
            _MODEL_BUFFER_TENSOR_MARKER: _MODEL_BUFFER_TENSOR_SCHEMA_VERSION,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data": data,
        }
    if isinstance(value, Mapping):
        return {
            key: _serialize_model_buffer_value(item, stats)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_model_buffer_value(item, stats) for item in value]
    if isinstance(value, tuple):
        return [_serialize_model_buffer_value(item, stats) for item in value]
    return value


def serialize_model_intermediate_buffer(
    buffer: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prepare a runner-owned payload for vLLM's typed request transport.

    ``OmniEngineCoreRequest.model_intermediate_buffer`` permits arbitrary
    nested metadata. Its msgspec field therefore decodes values as ``Any`` and
    cannot reliably infer that a native tensor representation must be restored
    to ``torch.Tensor``. Only tensors are converted to an explicit wire
    envelope; ordinary containers and scalar values keep their representation.
    """
    if buffer is None:
        return None
    if not isinstance(buffer, dict):
        raise TypeError(
            "model_intermediate_buffer must be a dictionary or None; "
            f"received {type(buffer).__name__}"
        )
    started_ns = time.perf_counter_ns()
    stats = _OperationStats()
    result = _serialize_model_buffer_value(buffer, stats)
    elapsed_ns = time.perf_counter_ns() - started_ns
    _record_handoff("serialize", elapsed_ns=elapsed_ns, stats=stats)
    _emit_handoff_event(
        "serialize",
        buffer=buffer,
        elapsed_ns=elapsed_ns,
        stats=stats,
    )
    return result


def _decode_model_buffer_tensor(
    value: Mapping[str, Any],
    stats: _OperationStats,
) -> torch.Tensor:
    dtype_name = value["dtype"]
    shape = value["shape"]
    data = value["data"]
    if not isinstance(dtype_name, str):
        raise TypeError("model_intermediate_buffer tensor dtype must be a string")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported model_intermediate_buffer tensor dtype: {dtype_name!r}")
    if not isinstance(shape, (list, tuple)) or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ValueError(f"Invalid model_intermediate_buffer tensor shape: {shape!r}")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("model_intermediate_buffer tensor data must be bytes-like")

    numel = math.prod(shape) if shape else 1
    expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected_bytes:
        raise ValueError(
            "model_intermediate_buffer tensor byte-size mismatch: "
            f"dtype={dtype_name} shape={tuple(shape)} "
            f"expected={expected_bytes} actual={len(data)}"
        )
    stats.tensor_count += 1
    stats.payload_bytes += len(data)
    if expected_bytes == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    owned_data = bytearray(data)
    return torch.frombuffer(owned_data, dtype=dtype).reshape(tuple(shape))


def _deserialize_model_buffer_value(value: Any, stats: _OperationStats) -> Any:
    if _is_model_buffer_tensor_envelope(value):
        return _decode_model_buffer_tensor(value, stats)
    if isinstance(value, Mapping):
        return {
            key: _deserialize_model_buffer_value(item, stats)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_deserialize_model_buffer_value(item, stats) for item in value]
    if isinstance(value, tuple):
        return [_deserialize_model_buffer_value(item, stats) for item in value]
    return value


def deserialize_model_intermediate_buffer(
    buffer: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Restore explicitly enveloped tensors after EngineCore IPC decoding."""
    if buffer is None:
        return None
    if not isinstance(buffer, dict):
        raise TypeError(
            "model_intermediate_buffer must be a dictionary or None; "
            f"received {type(buffer).__name__}"
        )
    started_ns = time.perf_counter_ns()
    stats = _OperationStats()
    result = _deserialize_model_buffer_value(buffer, stats)
    elapsed_ns = time.perf_counter_ns() - started_ns
    _record_handoff("deserialize", elapsed_ns=elapsed_ns, stats=stats)
    _emit_handoff_event(
        "deserialize",
        buffer=buffer,
        elapsed_ns=elapsed_ns,
        stats=stats,
    )
    return result


def serialize_additional_information(
    raw_info: dict[str, Any] | AdditionalInformationPayload | None,
    *,
    log_prefix: str | None = None,
) -> AdditionalInformationPayload | None:
    """Serialize omni request metadata for EngineCore transport.

    Delegates to ``serialize_payload`` which understands the nested
    ``OmniPayload`` TypedDict structure.
    """
    if raw_info is None:
        return None
    if isinstance(raw_info, AdditionalInformationPayload):
        return raw_info

    payload: OmniPayload = raw_info  # type: ignore[assignment]
    return serialize_payload(payload)


def deserialize_additional_information(
    payload: dict | AdditionalInformationPayload | None,
) -> dict:
    """Deserialize an *additional_information* payload into a plain dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    return deserialize_payload(payload)  # type: ignore[return-value]
