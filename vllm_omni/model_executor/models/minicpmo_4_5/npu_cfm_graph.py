# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, exact-shape NPU Graph cache for MiniCPM-o CFM decode."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_CFMOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_EagerCFM = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None],
    _CFMOutputs,
]


class NPUCFMGraphFallbackReason(str, Enum):
    """Stable, enumerable reasons for using the eager CFM implementation."""

    DISABLED = "disabled"
    UNSUPPORTED_DEVICE = "unsupported_device"
    UNSUPPORTED_API = "unsupported_api"
    CAPTURE_IN_PROGRESS = "capture_in_progress"
    EAGER_ONLY_KEY = "eager_only_key"
    BUDGET_REJECTION = "budget_rejection"
    CAPTURE_FAILURE = "capture_failure"
    REPLAY_FAILURE = "replay_failure"


@dataclass(frozen=True)
class TensorGraphSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> TensorGraphSignature:
        return cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=str(tensor.dtype),
            device_type=tensor.device.type,
            device_index=tensor.device.index,
        )


@dataclass(frozen=True)
class NPUCFMGraphKey:
    """All execution state that can change the captured tensor subgraph."""

    model_instance: int
    phase: str
    device_type: str
    device_index: int
    stream: tuple[str, str]
    batch_size: int
    source_tokens: TensorGraphSignature
    mu: TensorGraphSignature
    speakers: TensorGraphSignature
    cond: TensorGraphSignature
    cnn_cache: TensorGraphSignature | None
    att_cache: TensorGraphSignature | None
    steps: int
    last_chunk: bool
    flush_encoder: bool


@dataclass
class _GraphEntry:
    graph: Any
    static_inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]
    static_outputs: _CFMOutputs
    resident_bytes: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    failed: bool = False


class TorchNPUGraphRuntime:
    """Small capability boundary around the public ``torch.npu`` Graph API."""

    @staticmethod
    def _npu() -> Any | None:
        return getattr(torch, "npu", None)

    def api_supported(self) -> bool:
        npu = self._npu()
        required = (
            "NPUGraph",
            "current_device",
            "current_stream",
            "graph",
            "is_current_stream_capturing",
            "synchronize",
        )
        return npu is not None and all(callable(getattr(npu, name, None)) for name in required)

    @staticmethod
    def supports_device(device: torch.device) -> bool:
        return device.type == "npu"

    def device_index(self, device: torch.device) -> int:
        if device.index is not None:
            return int(device.index)
        npu = self._npu()
        if npu is None:
            raise RuntimeError("torch.npu is unavailable")
        return int(npu.current_device())

    def current_stream_identity(self, device: torch.device) -> tuple[str, str]:
        npu = self._npu()
        if npu is None:
            raise RuntimeError("torch.npu is unavailable")
        try:
            stream = npu.current_stream(device)
        except TypeError:
            stream = npu.current_stream()
        value = getattr(stream, "npu_stream", None)
        if value is None:
            value = getattr(stream, "stream_id", None)
        if value is None:
            value = id(stream)
        return type(stream).__name__, str(value)

    def is_current_stream_capturing(self) -> bool:
        npu = self._npu()
        if npu is None:
            return False
        return bool(npu.is_current_stream_capturing())

    def warmup(self, call: Callable[[], _CFMOutputs], device: torch.device) -> None:
        """Prime lazy kernels and allocations outside graph capture."""
        with torch.inference_mode():
            outputs = call()
        del outputs
        npu = self._npu()
        if npu is None:
            raise RuntimeError("torch.npu is unavailable")
        npu.synchronize(device)

    def new_graph(self) -> Any:
        npu = self._npu()
        if npu is None:
            raise RuntimeError("torch.npu is unavailable")
        return npu.NPUGraph()

    def capture(self, graph: Any, call: Callable[[], _CFMOutputs]) -> _CFMOutputs:
        npu = self._npu()
        if npu is None:
            raise RuntimeError("torch.npu is unavailable")
        with torch.inference_mode(), npu.graph(graph):
            return call()

    @staticmethod
    def replay(graph: Any) -> None:
        graph.replay()

    @staticmethod
    def reset(graph: Any) -> None:
        graph.reset()

    @staticmethod
    def graph_resident_bytes(graph: Any) -> int:
        # torch-npu does not currently expose a portable graph-pool byte count.
        # Runtime adapters may report graph-owned bytes here when available.
        del graph
        return 0

    def memory_snapshot(self, device: torch.device) -> tuple[int, int] | None:
        """Read Host allocator counters without synchronizing the device."""
        npu = self._npu()
        allocated = getattr(npu, "memory_allocated", None) if npu is not None else None
        reserved = getattr(npu, "memory_reserved", None) if npu is not None else None
        if not callable(allocated) or not callable(reserved):
            return None
        try:
            return int(allocated(device)), int(reserved(device))
        except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
            return None


class NPUCFMGraphRunner:
    """Capture and replay request-isolated CFM graphs under two hard budgets.

    Capture is serialized per model. Calls arriving during capture wait for the
    capture region to close and then execute eagerly, so unrelated request work
    cannot leak into a globally captured NPU stream.
    """

    def __init__(
        self,
        *,
        model_instance: object,
        steps: int,
        enabled: bool = False,
        max_entries: int = 4,
        max_bytes: int = 536870912,
        runtime: Any | None = None,
        max_eager_only_keys: int = 256,
    ) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("MiniCPM-o NPU CFM Graph enabled must be a boolean")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("MiniCPM-o NPU CFM Graph max_entries must be a positive integer")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("MiniCPM-o NPU CFM Graph max_bytes must be a positive integer")
        if type(max_eager_only_keys) is not int or not 1 <= max_eager_only_keys <= 256:
            raise ValueError("MiniCPM-o NPU CFM Graph eager-only limit must be an integer in [1, 256]")

        self.enabled = enabled
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.steps = int(steps)
        self._model_instance = id(model_instance)
        self._runtime = runtime if runtime is not None else TorchNPUGraphRuntime()
        self._max_eager_only_keys = max_eager_only_keys
        self._cache: OrderedDict[NPUCFMGraphKey, _GraphEntry] = OrderedDict()
        self._cache_bytes = 0
        self._eager_only: OrderedDict[NPUCFMGraphKey, None] = OrderedDict()
        self._capturing_keys: set[NPUCFMGraphKey] = set()
        self._metadata_lock = threading.RLock()
        self._capture_finished = threading.Condition(self._metadata_lock)
        self._stats = {
            "calls": 0,
            "eligible_calls": 0,
            "hits": 0,
            "misses": 0,
            "captures": 0,
            "capture_failures": 0,
            "eager_fallbacks": 0,
            "budget_rejections": 0,
            "evictions": 0,
        }
        self._fallback_reasons = {reason.value: 0 for reason in NPUCFMGraphFallbackReason}

    def telemetry(self) -> dict[str, int | dict[str, int]]:
        """Return fixed Host-only counters without querying or syncing the NPU."""
        with self._metadata_lock:
            return {
                **self._stats,
                "cache_entries": len(self._cache),
                "cache_bytes": self._cache_bytes,
                "eager_only_keys": len(self._eager_only),
                "fallback_reasons": dict(self._fallback_reasons),
            }

    @staticmethod
    def _call_eager(
        eager_fn: _EagerCFM,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ) -> _CFMOutputs:
        return eager_fn(*inputs)

    def _record_fallback_locked(self, reason: NPUCFMGraphFallbackReason) -> None:
        self._stats["eager_fallbacks"] += 1
        self._fallback_reasons[reason.value] += 1

    def _eager_fallback(
        self,
        reason: NPUCFMGraphFallbackReason,
        eager_fn: _EagerCFM,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ) -> _CFMOutputs:
        with self._metadata_lock:
            self._record_fallback_locked(reason)
        return self._call_eager(eager_fn, inputs)

    def _mark_eager_only_locked(self, key: NPUCFMGraphKey) -> None:
        self._eager_only[key] = None
        self._eager_only.move_to_end(key)
        while len(self._eager_only) > self._max_eager_only_keys:
            self._eager_only.popitem(last=False)

    def _safe_reset(self, graph: Any, *, key: NPUCFMGraphKey) -> None:
        try:
            self._runtime.reset(graph)
        except Exception:
            logger.warning("Failed to reset MiniCPM-o NPU CFM Graph for key %s", key, exc_info=True)

    @staticmethod
    def _clone_input(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        clone = torch.empty_strided(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        clone.copy_(tensor)
        return clone

    @classmethod
    def _static_inputs(
        cls,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        values = tuple(cls._clone_input(value) for value in inputs)
        assert values[0] is not None and values[1] is not None and values[2] is not None
        return values  # type: ignore[return-value]

    @staticmethod
    def _copy_inputs(
        destinations: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        sources: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ) -> None:
        for destination, source in zip(destinations, sources, strict=True):
            if destination is None or source is None:
                if destination is not source:
                    raise RuntimeError("NPU CFM Graph cache presence changed for an existing key")
                continue
            destination.copy_(source)

    @staticmethod
    def _validate_outputs(outputs: Any) -> _CFMOutputs:
        if (
            not isinstance(outputs, tuple)
            or len(outputs) != 3
            or not all(isinstance(output, torch.Tensor) for output in outputs)
        ):
            raise TypeError("MiniCPM-o NPU CFM Graph requires exactly three tensor outputs")
        return outputs

    @staticmethod
    def _copy_outputs(outputs: _CFMOutputs) -> _CFMOutputs:
        # Every request owns these tensors. Static graph outputs never enter the
        # Code2Wav request state and cannot be overwritten by a later replay.
        return tuple(output.clone() for output in outputs)  # type: ignore[return-value]

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor | None) -> int:
        return 0 if tensor is None else int(tensor.numel()) * int(tensor.element_size())

    def _resident_bytes(
        self,
        graph: Any,
        static_inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        static_outputs: _CFMOutputs,
        memory_before: tuple[int, int] | None,
        memory_after: tuple[int, int] | None,
    ) -> int:
        tensors = (*static_inputs, *static_outputs)
        tensor_bytes = sum(self._tensor_bytes(tensor) for tensor in tensors)
        graph_bytes = max(0, int(self._runtime.graph_resident_bytes(graph)))
        observed_delta = 0
        if memory_before is not None and memory_after is not None:
            allocated_delta = max(0, memory_after[0] - memory_before[0])
            reserved_delta = max(0, memory_after[1] - memory_before[1])
            observed_delta = max(allocated_delta, reserved_delta)
        # The allocator delta already includes static buffers when the backend
        # exposes it. Otherwise, fall back to explicit static bytes plus any
        # graph-owned byte estimate supplied by the runtime adapter.
        return max(observed_delta, tensor_bytes + graph_bytes)

    def _memory_snapshot(self, device: torch.device) -> tuple[int, int] | None:
        snapshot = getattr(self._runtime, "memory_snapshot", None)
        if not callable(snapshot):
            return None
        return snapshot(device)

    def _evict_one_locked(self) -> None:
        key, entry = self._cache.popitem(last=False)
        entry.lock.acquire()
        try:
            entry.failed = True
            self._safe_reset(entry.graph, key=key)
        finally:
            entry.lock.release()
        self._cache_bytes = max(0, self._cache_bytes - entry.resident_bytes)
        self._stats["evictions"] += 1

    def _make_key(
        self,
        *,
        phase: str,
        source_tokens: torch.Tensor,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        last_chunk: bool,
        flush_encoder: bool,
    ) -> NPUCFMGraphKey:
        if phase not in {"prompt", "stream"}:
            raise ValueError(f"MiniCPM-o NPU CFM Graph phase must be prompt or stream; got {phase!r}")
        mu, speakers, cond, cnn_cache, att_cache = inputs
        return NPUCFMGraphKey(
            model_instance=self._model_instance,
            phase=phase,
            device_type=mu.device.type,
            device_index=self._runtime.device_index(mu.device),
            stream=self._runtime.current_stream_identity(mu.device),
            batch_size=int(mu.shape[0]),
            source_tokens=TensorGraphSignature.from_tensor(source_tokens),
            mu=TensorGraphSignature.from_tensor(mu),
            speakers=TensorGraphSignature.from_tensor(speakers),
            cond=TensorGraphSignature.from_tensor(cond),
            cnn_cache=None if cnn_cache is None else TensorGraphSignature.from_tensor(cnn_cache),
            att_cache=None if att_cache is None else TensorGraphSignature.from_tensor(att_cache),
            steps=self.steps,
            last_chunk=bool(last_chunk),
            flush_encoder=bool(flush_encoder),
        )

    def _discard_failed_entry(self, key: NPUCFMGraphKey, entry: _GraphEntry, *, eager_only: bool) -> None:
        with self._metadata_lock:
            if self._cache.get(key) is entry:
                self._cache.pop(key)
                self._cache_bytes = max(0, self._cache_bytes - entry.resident_bytes)
            if eager_only:
                self._mark_eager_only_locked(key)

    def _replay_entry(
        self,
        key: NPUCFMGraphKey,
        entry: _GraphEntry,
        inputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        eager_fn: _EagerCFM,
    ) -> _CFMOutputs:
        failure: BaseException | None = None
        output: _CFMOutputs | None = None
        try:
            if entry.failed:
                raise RuntimeError("NPU CFM Graph entry was reset")
            self._copy_inputs(entry.static_inputs, inputs)
            self._runtime.replay(entry.graph)
            output = self._copy_outputs(entry.static_outputs)
        except BaseException as error:
            failure = error
            entry.failed = True
            self._safe_reset(entry.graph, key=key)
        finally:
            entry.lock.release()

        if failure is None:
            assert output is not None
            return output

        self._discard_failed_entry(key, entry, eager_only=isinstance(failure, Exception))
        if not isinstance(failure, Exception):
            raise failure
        logger.warning(
            "MiniCPM-o NPU CFM Graph replay failed for key %s; using eager for this key (%s: %s)",
            key,
            type(failure).__name__,
            failure,
        )
        return self._eager_fallback(NPUCFMGraphFallbackReason.REPLAY_FAILURE, eager_fn, inputs)

    def run(
        self,
        *,
        eager_fn: _EagerCFM,
        source_tokens: torch.Tensor,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        phase: str,
        last_chunk: bool,
        flush_encoder: bool,
    ) -> _CFMOutputs:
        inputs = (mu, speakers, cond, cnn_cache, att_cache)
        with self._metadata_lock:
            self._stats["calls"] += 1

        if not self.enabled:
            return self._eager_fallback(NPUCFMGraphFallbackReason.DISABLED, eager_fn, inputs)
        if not self._runtime.supports_device(mu.device):
            return self._eager_fallback(NPUCFMGraphFallbackReason.UNSUPPORTED_DEVICE, eager_fn, inputs)
        if not self._runtime.api_supported():
            logger.warning_once("MiniCPM-o NPU CFM Graph requested but the torch.npu Graph API is unavailable")
            return self._eager_fallback(NPUCFMGraphFallbackReason.UNSUPPORTED_API, eager_fn, inputs)

        try:
            key = self._make_key(
                phase=phase,
                source_tokens=source_tokens,
                inputs=inputs,
                last_chunk=last_chunk,
                flush_encoder=flush_encoder,
            )
            stream_is_capturing = self._runtime.is_current_stream_capturing()
        except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError) as error:
            logger.warning_once(
                "MiniCPM-o NPU CFM Graph requested but runtime capability discovery failed; using eager (%s: %s)",
                type(error).__name__,
                error,
            )
            return self._eager_fallback(NPUCFMGraphFallbackReason.UNSUPPORTED_API, eager_fn, inputs)

        with self._metadata_lock:
            self._stats["eligible_calls"] += 1
            if stream_is_capturing:
                self._stats["misses"] += 1
                self._record_fallback_locked(NPUCFMGraphFallbackReason.CAPTURE_IN_PROGRESS)
                run_eager = True
            elif self._capturing_keys:
                self._stats["misses"] += 1
                self._record_fallback_locked(NPUCFMGraphFallbackReason.CAPTURE_IN_PROGRESS)
                while self._capturing_keys:
                    self._capture_finished.wait()
                run_eager = True
            else:
                run_eager = False

            if run_eager:
                entry = None
            else:
                entry = self._cache.get(key)
                if entry is not None:
                    self._cache.move_to_end(key)
                    self._stats["hits"] += 1
                    entry.lock.acquire()
                else:
                    self._stats["misses"] += 1
                    if key in self._eager_only:
                        self._eager_only.move_to_end(key)
                        self._record_fallback_locked(NPUCFMGraphFallbackReason.EAGER_ONLY_KEY)
                        run_eager = True
                    else:
                        self._capturing_keys.add(key)

        if run_eager:
            return self._call_eager(eager_fn, inputs)
        if entry is not None:
            return self._replay_entry(key, entry, inputs, eager_fn)

        graph: Any | None = None
        try:
            memory_before = self._memory_snapshot(mu.device)
            static_inputs = self._static_inputs(inputs)
            self._runtime.warmup(
                lambda: self._call_eager(eager_fn, static_inputs),
                mu.device,
            )
            graph = self._runtime.new_graph()
            static_outputs = self._validate_outputs(
                self._runtime.capture(
                    graph,
                    lambda: self._call_eager(eager_fn, static_inputs),
                )
            )
            # Capture records kernels but does not guarantee that output buffers
            # contain the first request's result. Replay once before copy-out.
            self._runtime.replay(graph)
            memory_after = self._memory_snapshot(mu.device)
            resident_bytes = self._resident_bytes(
                graph,
                static_inputs,
                static_outputs,
                memory_before,
                memory_after,
            )
            first_output = self._copy_outputs(static_outputs)
            captured_entry = _GraphEntry(
                graph=graph,
                static_inputs=static_inputs,
                static_outputs=static_outputs,
                resident_bytes=resident_bytes,
            )
        except BaseException as error:
            if graph is not None:
                self._safe_reset(graph, key=key)
            with self._metadata_lock:
                self._capturing_keys.discard(key)
                self._capture_finished.notify_all()
                if isinstance(error, Exception):
                    self._stats["capture_failures"] += 1
                    self._mark_eager_only_locked(key)
            if not isinstance(error, Exception):
                raise
            logger.warning(
                "MiniCPM-o NPU CFM Graph capture failed for key %s; using eager (%s: %s)",
                key,
                type(error).__name__,
                error,
            )
            return self._eager_fallback(NPUCFMGraphFallbackReason.CAPTURE_FAILURE, eager_fn, inputs)

        reject_budget = False
        with self._metadata_lock:
            if captured_entry.resident_bytes > self.max_bytes:
                self._stats["budget_rejections"] += 1
                self._mark_eager_only_locked(key)
                reject_budget = True
            else:
                while self._cache and (
                    len(self._cache) >= self.max_entries
                    or self._cache_bytes + captured_entry.resident_bytes > self.max_bytes
                ):
                    self._evict_one_locked()
                if self._cache_bytes + captured_entry.resident_bytes > self.max_bytes:
                    self._stats["budget_rejections"] += 1
                    self._mark_eager_only_locked(key)
                    reject_budget = True
                else:
                    self._cache[key] = captured_entry
                    self._cache_bytes += captured_entry.resident_bytes
                    self._stats["captures"] += 1
            if not reject_budget:
                self._capturing_keys.discard(key)
                self._capture_finished.notify_all()

        if reject_budget:
            captured_entry.failed = True
            self._safe_reset(captured_entry.graph, key=key)
            with self._metadata_lock:
                self._capturing_keys.discard(key)
                self._capture_finished.notify_all()
            return self._eager_fallback(NPUCFMGraphFallbackReason.BUDGET_REJECTION, eager_fn, inputs)
        return first_output
