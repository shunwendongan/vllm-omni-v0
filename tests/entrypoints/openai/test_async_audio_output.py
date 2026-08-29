# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import io
import threading
import wave

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai.async_audio_output import (
    ASYNC_AUDIO_OUTPUT_ENV,
    AsyncAudioOutputPipeline,
    AudioOutputPipelineError,
    AudioOutputPipelineStaleEpochError,
    async_audio_output_enabled,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("value", [True, 1, "1", "true", " yes ", "on"])
def test_async_audio_output_accepts_explicit_true(value):
    assert async_audio_output_enabled(value) is True


@pytest.mark.parametrize("value", [False, 0, "0", "false", " no ", "off"])
def test_async_audio_output_accepts_explicit_false(value):
    assert async_audio_output_enabled(value) is False


@pytest.mark.parametrize("value", [2, -1, "", "enabled", object()])
def test_async_audio_output_rejects_invalid_values(value):
    with pytest.raises(ValueError, match=ASYNC_AUDIO_OUTPUT_ENV):
        async_audio_output_enabled(value)


@pytest.mark.asyncio
async def test_two_slot_ring_is_bounded_and_delivers_in_order():
    pipeline = AsyncAudioOutputPipeline(max_pending=4)
    release = threading.Event()

    def first_encoder(audio):
        release.wait(timeout=5)
        return float(audio[0])

    first = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=0,
        audio=torch.tensor([1.0]),
        encoder=first_encoder,
    )
    second = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=1,
        audio=torch.tensor([2.0]),
        encoder=lambda audio: float(audio[0]),
    )
    with pytest.raises(AudioOutputPipelineError, match="ring is full"):
        pipeline.submit(
            request_id="req",
            epoch=0,
            sequence=2,
            audio=torch.tensor([3.0]),
            encoder=lambda audio: float(audio[0]),
        )
    with pytest.raises(AudioOutputPipelineError, match="expected delivery 0"):
        await pipeline.resolve(second)

    release.set()
    assert await pipeline.resolve(first) == 1.0
    assert await pipeline.resolve(second) == 2.0
    pipeline.finish("req", epoch=0)
    assert pipeline.active_requests == 0
    pipeline.close()


@pytest.mark.asyncio
async def test_completed_encoding_keeps_slot_until_ordered_publication():
    pipeline = AsyncAudioOutputPipeline(max_pending=4)
    first = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=0,
        audio=torch.tensor([1.0]),
        encoder=lambda audio: float(audio[0]),
    )
    second = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=1,
        audio=torch.tensor([2.0]),
        encoder=lambda audio: float(audio[0]),
    )
    assert first.future.result(timeout=5) == 1.0
    with pytest.raises(AudioOutputPipelineError, match="ring is full"):
        pipeline.submit(
            request_id="req",
            epoch=0,
            sequence=2,
            audio=torch.tensor([3.0]),
            encoder=lambda audio: float(audio[0]),
        )

    assert await pipeline.resolve(first) == 1.0
    third = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=2,
        audio=torch.tensor([3.0]),
        encoder=lambda audio: float(audio[0]),
    )
    assert await pipeline.resolve(second) == 2.0
    assert await pipeline.resolve(third) == 3.0
    pipeline.finish("req", epoch=0)
    pipeline.close()


@pytest.mark.asyncio
async def test_request_rings_do_not_mix_audio_or_sequence_state():
    pipeline = AsyncAudioOutputPipeline(max_pending=4)
    left = pipeline.submit(
        request_id="left",
        epoch=0,
        sequence=0,
        audio=torch.tensor([11.0]),
        encoder=lambda audio: ("left", float(audio[0])),
    )
    right = pipeline.submit(
        request_id="right",
        epoch=7,
        sequence=0,
        audio=torch.tensor([22.0]),
        encoder=lambda audio: ("right", float(audio[0])),
    )

    assert await pipeline.resolve(right) == ("right", 22.0)
    assert await pipeline.resolve(left) == ("left", 11.0)
    pipeline.finish("right", epoch=7)
    pipeline.finish("left", epoch=0)
    assert pipeline.active_requests == 0
    pipeline.close()


@pytest.mark.asyncio
async def test_finish_rejects_pending_tail_without_releasing_ring():
    pipeline = AsyncAudioOutputPipeline()
    ticket = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=0,
        audio=torch.tensor([1.0]),
        encoder=lambda audio: float(audio[0]),
    )
    with pytest.raises(AudioOutputPipelineError, match="pending slots"):
        pipeline.finish("req", epoch=0)
    assert pipeline.active_requests == 1
    assert await pipeline.resolve(ticket) == 1.0
    pipeline.finish("req", epoch=0)
    assert pipeline.active_requests == 0
    pipeline.close()


@pytest.mark.asyncio
async def test_epoch_change_isolates_old_ticket_and_abort_releases_state():
    pipeline = AsyncAudioOutputPipeline(max_pending=4)
    release = threading.Event()
    old = pipeline.submit(
        request_id="req",
        epoch=2,
        sequence=0,
        audio=torch.tensor([1.0]),
        encoder=lambda audio: release.wait(timeout=5),
    )
    new = pipeline.submit(
        request_id="req",
        epoch=3,
        sequence=0,
        audio=torch.tensor([3.0]),
        encoder=lambda audio: float(audio[0]),
    )
    release.set()
    with pytest.raises(AudioOutputPipelineStaleEpochError):
        await pipeline.resolve(old)
    assert await pipeline.resolve(new) == 3.0
    pipeline.abort("req", epoch=3)
    assert pipeline.active_requests == 0
    pipeline.close()


@pytest.mark.asyncio
async def test_encoder_exception_aborts_request_without_slot_leak():
    pipeline = AsyncAudioOutputPipeline()

    def fail(audio):
        raise RuntimeError(f"bad audio {audio.size}")

    ticket = pipeline.submit(
        request_id="req",
        epoch=0,
        sequence=0,
        audio=torch.tensor([1.0]),
        encoder=fail,
    )
    with pytest.raises(RuntimeError, match="bad audio"):
        await pipeline.resolve(ticket)
    assert pipeline.active_requests == 0
    pipeline.close()


@pytest.mark.asyncio
async def test_pcm_wav_and_base64_bytes_are_decodable():
    pipeline = AsyncAudioOutputPipeline()

    def encode_wav(audio: np.ndarray) -> str:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    encoded = await pipeline.encode(
        request_id="req",
        epoch=0,
        sequence=0,
        audio=torch.tensor([0.0, 0.5, -0.5]),
        encoder=encode_wav,
    )
    with wave.open(io.BytesIO(base64.b64decode(encoded)), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 3
    pipeline.finish("req", epoch=0)
    pipeline.close()
