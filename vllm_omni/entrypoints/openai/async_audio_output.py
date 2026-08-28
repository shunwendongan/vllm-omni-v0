# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, ordered audio staging and encoding for streaming responses."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np
import torch

ASYNC_AUDIO_OUTPUT_ENV = "VLLM_OMNI_MINICPMO45_ASYNC_AUDIO_OUTPUT"

_T = TypeVar("_T")


class AudioOutputPipelineError(RuntimeError):
    pass


class AudioOutputPipelineUnsupportedError(AudioOutputPipelineError):
    pass


class AudioOutputPipelineStaleEpochError(AudioOutputPipelineError):
    pass


def async_audio_output_enabled(value: object | None = None) -> bool:
    """Parse the opt-in environment flag without accepting ambiguous values."""
    raw = os.environ.get(ASYNC_AUDIO_OUTPUT_ENV, "0") if value is None else value
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{ASYNC_AUDIO_OUTPUT_ENV} must be an explicit boolean; got {raw!r}")


@dataclass
class _StagedAudio:
    host_tensor: torch.Tensor
    ready_event: Any | None
    source_tensor: torch.Tensor | None
    pinned: bool

    def numpy(self) -> np.ndarray:
        if self.ready_event is not None:
            self.ready_event.synchronize()
        self.source_tensor = None
        return self.host_tensor.numpy()


@dataclass
class _Slot(Generic[_T]):
    generation: int = 0
    sequence: int | None = None
    future: Future[_T] | None = None


@dataclass
class _RequestRing(Generic[_T]):
    epoch: int
    next_submit: int
    next_deliver: int
    slots: list[_Slot[_T]] = field(default_factory=lambda: [_Slot(), _Slot()])


@dataclass(frozen=True)
class AudioOutputTicket(Generic[_T]):
    request_id: str
    epoch: int
    sequence: int
    slot_index: int
    generation: int
    future: Future[_T]


class AsyncAudioOutputPipeline:
    """Request-owned two-slot ring backed by one bounded encoding worker."""

    def __init__(self, *, max_pending: int = 32) -> None:
        if max_pending < 2:
            raise ValueError("max_pending must be at least two")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omni-audio-output")
        self._capacity = threading.BoundedSemaphore(max_pending)
        self._lock = threading.RLock()
        self._requests: dict[str, _RequestRing[Any]] = {}
        self._copy_streams: dict[str, Any] = {}
        self._closed = False

    @property
    def active_requests(self) -> int:
        with self._lock:
            return len(self._requests)

    def _pinned_host_like(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        try:
            return (
                torch.empty_like(tensor, dtype=torch.float32, device="cpu", pin_memory=True),
                True,
            )
        except (RuntimeError, TypeError):
            if tensor.device.type != "cpu":
                raise AudioOutputPipelineUnsupportedError("pinned host allocation is unavailable") from None
            return torch.empty_like(tensor, dtype=torch.float32, device="cpu"), False

    def _stage(self, tensor: torch.Tensor) -> _StagedAudio:
        source = tensor.detach().contiguous()
        host, pinned = self._pinned_host_like(source)
        if source.device.type == "cpu":
            host.copy_(source)
            return _StagedAudio(host, None, None, pinned)
        if source.device.type != "npu":
            raise AudioOutputPipelineUnsupportedError(f"unsupported audio device: {source.device.type}")

        npu = getattr(torch, "npu", None)
        if npu is None or not all(hasattr(npu, name) for name in ("Event", "Stream", "current_stream", "stream")):
            raise AudioOutputPipelineUnsupportedError("torch-npu copy stream APIs are unavailable")
        device_key = str(source.device)
        copy_stream = self._copy_streams.get(device_key)
        if copy_stream is None:
            copy_stream = npu.Stream(device=source.device)
            self._copy_streams[device_key] = copy_stream
        ready_event = npu.Event()
        with npu.stream(copy_stream):
            copy_stream.wait_stream(npu.current_stream(source.device))
            host.copy_(source, non_blocking=True)
            ready_event.record()
        return _StagedAudio(host, ready_event, source, pinned)

    @staticmethod
    def _encode(staged: _StagedAudio, encoder: Callable[[np.ndarray], _T]) -> _T:
        return encoder(staged.numpy())

    def submit(
        self,
        *,
        request_id: str,
        epoch: int,
        sequence: int,
        audio: torch.Tensor,
        encoder: Callable[[np.ndarray], _T],
    ) -> AudioOutputTicket[_T]:
        """Stage one chunk; callers must resolve tickets in sequence order."""
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if epoch < 0 or sequence < 0:
            raise ValueError("epoch and sequence must be non-negative")
        if audio.numel() == 0:
            raise ValueError("empty audio chunks must bypass the async pipeline")
        with self._lock:
            if self._closed:
                raise AudioOutputPipelineError("audio output pipeline is closed")
            ring = self._requests.get(request_id)
            if ring is None:
                ring = _RequestRing(epoch=epoch, next_submit=sequence, next_deliver=sequence)
                self._requests[request_id] = ring
            elif epoch < ring.epoch:
                raise AudioOutputPipelineStaleEpochError(
                    f"request {request_id} epoch {epoch} is older than active epoch {ring.epoch}"
                )
            elif epoch > ring.epoch:
                self._abort_locked(request_id)
                ring = _RequestRing(epoch=epoch, next_submit=sequence, next_deliver=sequence)
                self._requests[request_id] = ring
            if sequence != ring.next_submit:
                raise AudioOutputPipelineError(
                    f"request {request_id} expected sequence {ring.next_submit}, got {sequence}"
                )
            slot_index = sequence % len(ring.slots)
            slot = ring.slots[slot_index]
            if slot.future is not None and not slot.future.done():
                raise AudioOutputPipelineError(f"request {request_id} audio staging ring is full")
            if not self._capacity.acquire(blocking=False):
                raise AudioOutputPipelineError("global audio encoding queue is full")
            try:
                staged = self._stage(audio)
                future = self._executor.submit(self._encode, staged, encoder)
            except BaseException:
                self._capacity.release()
                raise
            future.add_done_callback(lambda _: self._capacity.release())
            slot.generation += 1
            slot.sequence = sequence
            slot.future = future
            ring.next_submit += 1
            return AudioOutputTicket(
                request_id=request_id,
                epoch=epoch,
                sequence=sequence,
                slot_index=slot_index,
                generation=slot.generation,
                future=future,
            )

    async def resolve(self, ticket: AudioOutputTicket[_T]) -> _T:
        with self._lock:
            ring = self._requests.get(ticket.request_id)
            if ring is None or ring.epoch != ticket.epoch:
                raise AudioOutputPipelineStaleEpochError(
                    f"request {ticket.request_id} epoch {ticket.epoch} is no longer active"
                )
            if ticket.sequence != ring.next_deliver:
                raise AudioOutputPipelineError(
                    f"request {ticket.request_id} expected delivery {ring.next_deliver}, got {ticket.sequence}"
                )
            slot = ring.slots[ticket.slot_index]
            if slot.generation != ticket.generation or slot.future is not ticket.future:
                raise AudioOutputPipelineStaleEpochError("audio staging slot was recycled")
        try:
            result = await asyncio.wrap_future(ticket.future)
        except BaseException:
            self.abort(ticket.request_id, epoch=ticket.epoch)
            raise
        with self._lock:
            ring = self._requests.get(ticket.request_id)
            if ring is None or ring.epoch != ticket.epoch:
                raise AudioOutputPipelineStaleEpochError("audio request was aborted while encoding")
            slot = ring.slots[ticket.slot_index]
            if slot.generation != ticket.generation:
                raise AudioOutputPipelineStaleEpochError("audio staging slot was recycled")
            slot.sequence = None
            slot.future = None
            ring.next_deliver += 1
        return result

    async def encode(
        self,
        *,
        request_id: str,
        epoch: int,
        sequence: int,
        audio: torch.Tensor,
        encoder: Callable[[np.ndarray], _T],
    ) -> _T:
        ticket = self.submit(
            request_id=request_id,
            epoch=epoch,
            sequence=sequence,
            audio=audio,
            encoder=encoder,
        )
        return await self.resolve(ticket)

    def _abort_locked(self, request_id: str) -> None:
        ring = self._requests.pop(request_id, None)
        if ring is None:
            return
        for slot in ring.slots:
            if slot.future is not None:
                slot.future.cancel()
            slot.future = None
            slot.sequence = None

    def abort(self, request_id: str, *, epoch: int | None = None) -> None:
        with self._lock:
            ring = self._requests.get(request_id)
            if ring is None or (epoch is not None and ring.epoch != epoch):
                return
            self._abort_locked(request_id)

    def finish(self, request_id: str, *, epoch: int) -> None:
        """Release an idle request ring after its ordered tail was published."""
        with self._lock:
            ring = self._requests.get(request_id)
            if ring is None or ring.epoch != epoch:
                return
            if any(slot.future is not None for slot in ring.slots):
                raise AudioOutputPipelineError("cannot finish audio request with pending slots")
            self._requests.pop(request_id, None)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for request_id in list(self._requests):
                self._abort_locked(request_id)
        self._executor.shutdown(wait=True, cancel_futures=True)
