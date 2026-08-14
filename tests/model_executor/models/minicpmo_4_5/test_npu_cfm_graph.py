# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.npu_cfm_graph import (
    NPUCFMGraphRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeGraph:
    def __init__(self) -> None:
        self.call = None
        self.outputs = None
        self.replays = 0
        self.resets = 0


class _CaptureCancelledError(BaseException):
    pass


class _FakeRuntime:
    def __init__(self, *, graph_bytes: int = 0, observed_capture_bytes: int = 0) -> None:
        self.supported = True
        self.api_available = True
        self.capturing = False
        self.stream = "stream-0"
        self.graph_bytes = graph_bytes
        self.observed_capture_bytes = observed_capture_bytes
        self.memory_bytes = 0
        self.graphs: list[_FakeGraph] = []
        self.warmups = 0
        self.fail_capture: BaseException | None = None
        self.fail_replay = False
        self.capture_started = threading.Event()
        self.capture_release: threading.Event | None = None

    def api_supported(self) -> bool:
        return self.api_available

    def supports_device(self, _device: torch.device) -> bool:
        return self.supported

    @staticmethod
    def device_index(_device: torch.device) -> int:
        return 0

    def current_stream_identity(self, _device: torch.device) -> tuple[str, str]:
        return "FakeStream", self.stream

    def is_current_stream_capturing(self) -> bool:
        return self.capturing

    def warmup(self, call, _device: torch.device) -> None:
        self.warmups += 1
        call()

    def new_graph(self) -> _FakeGraph:
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph

    def capture(self, graph: _FakeGraph, call):
        self.capture_started.set()
        if self.capture_release is not None:
            assert self.capture_release.wait(timeout=2)
        if self.fail_capture is not None:
            error = self.fail_capture
            self.fail_capture = None
            raise error
        graph.call = call
        graph.outputs = call()
        self.memory_bytes += self.observed_capture_bytes
        return graph.outputs

    def replay(self, graph: _FakeGraph) -> None:
        if self.fail_replay:
            raise RuntimeError("injected replay failure")
        assert graph.call is not None and graph.outputs is not None
        values = graph.call()
        for destination, source in zip(graph.outputs, values, strict=True):
            destination.copy_(source)
        graph.replays += 1

    @staticmethod
    def reset(graph: _FakeGraph) -> None:
        graph.resets += 1

    def graph_resident_bytes(self, _graph: _FakeGraph) -> int:
        return self.graph_bytes

    def memory_snapshot(self, _device: torch.device) -> tuple[int, int]:
        return self.memory_bytes, self.memory_bytes


def _eager_cfm(mu, speakers, cond, cnn_cache, att_cache):
    speaker_features = speakers.unsqueeze(-1).expand_as(mu)
    mel = mu + speaker_features + cond
    cnn = mu * 2 if cnn_cache is None else mu + cnn_cache
    att = cond * 3 if att_cache is None else cond + att_cache
    return mel, cnn, att


def _runner(
    runtime: _FakeRuntime,
    *,
    max_entries: int = 4,
    max_bytes: int = 1 << 30,
    max_eager_only_keys: int = 256,
) -> NPUCFMGraphRunner:
    return NPUCFMGraphRunner(
        model_instance=object(),
        steps=10,
        enabled=True,
        max_entries=max_entries,
        max_bytes=max_bytes,
        runtime=runtime,
        max_eager_only_keys=max_eager_only_keys,
    )


def _run(
    runner: NPUCFMGraphRunner,
    value: float,
    *,
    width: int = 1,
    token_width: int | None = None,
    phase: str = "stream",
    last_chunk: bool = False,
    flush_encoder: bool = False,
    with_cache: bool = False,
):
    mu = torch.full((1, 1, width), value)
    cond = torch.full_like(mu, value / 10)
    speakers = torch.full((1, 1), value / 100)
    source_tokens = torch.full((1, token_width or width), int(value), dtype=torch.long)
    cnn_cache = torch.full_like(mu, value / 1000) if with_cache else None
    att_cache = torch.full_like(mu, value / 10000) if with_cache else None
    return runner.run(
        eager_fn=_eager_cfm,
        source_tokens=source_tokens,
        mu=mu,
        speakers=speakers,
        cond=cond,
        cnn_cache=cnn_cache,
        att_cache=att_cache,
        phase=phase,
        last_chunk=last_chunk,
        flush_encoder=flush_encoder,
    )


def test_miss_capture_hit_copies_every_input_and_returns_owned_outputs():
    runtime = _FakeRuntime()
    runner = _runner(runtime)

    first = _run(runner, 1.0, with_cache=True)
    second = _run(runner, 7.0, with_cache=True)

    expected = _eager_cfm(
        torch.full((1, 1, 1), 7.0),
        torch.full((1, 1), 0.07),
        torch.full((1, 1, 1), 0.7),
        torch.full((1, 1, 1), 0.007),
        torch.full((1, 1, 1), 0.0007),
    )
    for actual, reference in zip(second, expected, strict=True):
        torch.testing.assert_close(actual, reference)

    entry = next(iter(runner._cache.values()))
    torch.testing.assert_close(entry.static_inputs[0], torch.full((1, 1, 1), 7.0))
    torch.testing.assert_close(entry.static_inputs[1], torch.full((1, 1), 0.07))
    torch.testing.assert_close(entry.static_inputs[2], torch.full((1, 1, 1), 0.7))
    torch.testing.assert_close(entry.static_inputs[3], torch.full((1, 1, 1), 0.007))
    torch.testing.assert_close(entry.static_inputs[4], torch.full((1, 1, 1), 0.0007))
    for first_output, second_output, static_output in zip(first, second, entry.static_outputs, strict=True):
        assert first_output.data_ptr() != second_output.data_ptr()
        assert first_output.data_ptr() != static_output.data_ptr()
        assert second_output.data_ptr() != static_output.data_ptr()

    telemetry = runner.telemetry()
    assert telemetry["calls"] == 2
    assert telemetry["eligible_calls"] == 2
    assert telemetry["misses"] == 1
    assert telemetry["captures"] == 1
    assert telemetry["hits"] == 1
    assert telemetry["eager_fallbacks"] == 0
    assert runtime.warmups == 1
    assert runtime.graphs[0].replays == 2


def test_key_covers_model_phase_stream_tensor_cache_steps_and_tail_flags():
    runtime = _FakeRuntime()
    model = object()
    runner = NPUCFMGraphRunner(
        model_instance=model,
        steps=8,
        enabled=True,
        runtime=runtime,
    )

    _run(runner, 2.0, token_width=3, phase="prompt", with_cache=True)
    key = next(iter(runner._cache))

    assert key.model_instance == id(model)
    assert key.phase == "prompt"
    assert key.device_type == "cpu"
    assert key.device_index == 0
    assert key.stream == ("FakeStream", "stream-0")
    assert key.batch_size == 1
    assert key.source_tokens.shape == (1, 3)
    assert key.source_tokens.stride == (3, 1)
    assert key.mu.shape == (1, 1, 1)
    assert key.mu.stride == (1, 1, 1)
    assert key.cnn_cache is not None
    assert key.att_cache is not None
    assert key.steps == 8
    assert key.last_chunk is False
    assert key.flush_encoder is False

    runtime.stream = "stream-1"
    _run(runner, 2.0, token_width=3, phase="stream", last_chunk=True, flush_encoder=True, with_cache=True)
    keys = list(runner._cache)
    assert len(keys) == 2
    assert keys[-1].stream == ("FakeStream", "stream-1")
    assert keys[-1].phase == "stream"
    assert keys[-1].last_chunk is True
    assert keys[-1].flush_encoder is True


def test_count_lru_evicts_true_least_recently_used_graph_and_resets_it():
    runtime = _FakeRuntime()
    runner = _runner(runtime, max_entries=2)

    _run(runner, 1.0, phase="prompt")  # A
    _run(runner, 2.0, phase="stream")  # B
    _run(runner, 3.0, phase="prompt")  # A hit; B becomes LRU
    _run(runner, 4.0, phase="stream", last_chunk=True)  # C

    assert runtime.graphs[0].resets == 0
    assert runtime.graphs[1].resets == 1
    assert runtime.graphs[2].resets == 0
    telemetry = runner.telemetry()
    assert telemetry["cache_entries"] == 2
    assert telemetry["evictions"] == 1
    assert telemetry["hits"] == 1


def test_byte_budget_evicts_and_resets_before_admitting_new_entry():
    runtime = _FakeRuntime(graph_bytes=100)
    runner = _runner(runtime, max_entries=4, max_bytes=130)

    _run(runner, 1.0, phase="prompt")
    assert runner.telemetry()["cache_bytes"] == 124
    _run(runner, 2.0, phase="stream")

    telemetry = runner.telemetry()
    assert telemetry["cache_entries"] == 1
    assert telemetry["cache_bytes"] == 124
    assert telemetry["evictions"] == 1
    assert runtime.graphs[0].resets == 1


def test_single_graph_over_budget_is_reset_and_key_becomes_eager_only():
    runtime = _FakeRuntime(graph_bytes=100)
    runner = _runner(runtime, max_bytes=100)

    first = _run(runner, 1.0)
    second = _run(runner, 2.0)

    torch.testing.assert_close(first[0], torch.tensor([[[1.11]]]))
    torch.testing.assert_close(second[0], torch.tensor([[[2.22]]]))
    telemetry = runner.telemetry()
    assert telemetry["captures"] == 0
    assert telemetry["budget_rejections"] == 1
    assert telemetry["cache_entries"] == 0
    assert telemetry["cache_bytes"] == 0
    assert telemetry["eager_only_keys"] == 1
    assert telemetry["eager_fallbacks"] == 2
    assert telemetry["fallback_reasons"]["budget_rejection"] == 1
    assert telemetry["fallback_reasons"]["eager_only_key"] == 1
    assert len(runtime.graphs) == 1
    assert runtime.graphs[0].resets == 1


def test_observed_allocator_delta_participates_in_resident_byte_budget():
    runtime = _FakeRuntime(observed_capture_bytes=200)
    runner = _runner(runtime, max_bytes=150)

    _run(runner, 1.0)

    telemetry = runner.telemetry()
    assert telemetry["budget_rejections"] == 1
    assert telemetry["cache_entries"] == 0
    assert runtime.graphs[0].resets == 1


def test_capture_failures_are_bounded_and_do_not_retry_retained_keys():
    runtime = _FakeRuntime()
    runner = _runner(runtime, max_eager_only_keys=2)

    for token_width in (1, 2, 3):
        runtime.fail_capture = RuntimeError("unsupported captured op")
        _run(runner, float(token_width), token_width=token_width)

    assert runner.telemetry()["eager_only_keys"] == 2
    graph_count = len(runtime.graphs)
    _run(runner, 2.0, token_width=2)
    assert len(runtime.graphs) == graph_count

    # Key 1 was the bounded set's LRU and may be attempted once more.
    runtime.fail_capture = RuntimeError("unsupported captured op")
    _run(runner, 1.0, token_width=1)
    telemetry = runner.telemetry()
    assert telemetry["capture_failures"] == 4
    assert telemetry["eager_only_keys"] == 2
    assert telemetry["fallback_reasons"]["capture_failure"] == 4
    assert telemetry["fallback_reasons"]["eager_only_key"] == 1
    assert all(graph.resets == 1 for graph in runtime.graphs)


def test_interleaved_mixed_buckets_and_repeated_replays_have_no_stale_state():
    runtime = _FakeRuntime()
    runner = _runner(runtime)

    a1 = _run(runner, 1.0, width=1)
    b = _run(runner, 5.0, width=2)
    a2 = _run(runner, 9.0, width=1)
    a3 = _run(runner, 3.0, width=1)

    torch.testing.assert_close(a1[0], torch.tensor([[[1.11]]]))
    torch.testing.assert_close(b[0], torch.full((1, 1, 2), 5.55))
    torch.testing.assert_close(a2[0], torch.tensor([[[9.99]]]))
    torch.testing.assert_close(a3[0], torch.tensor([[[3.33]]]))
    assert a1[0].data_ptr() != a2[0].data_ptr() != a3[0].data_ptr()
    telemetry = runner.telemetry()
    assert telemetry["captures"] == 2
    assert telemetry["hits"] == 2
    assert telemetry["cache_entries"] == 2


def test_calls_during_capture_wait_for_cleanup_then_take_eager_fallback():
    runtime = _FakeRuntime()
    runtime.capture_release = threading.Event()
    runner = _runner(runtime)
    outputs: dict[str, tuple[torch.Tensor, ...]] = {}

    first = threading.Thread(target=lambda: outputs.setdefault("first", _run(runner, 1.0)))
    second = threading.Thread(target=lambda: outputs.setdefault("second", _run(runner, 2.0)))
    first.start()
    assert runtime.capture_started.wait(timeout=2)
    second.start()
    assert second.is_alive()
    runtime.capture_release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    torch.testing.assert_close(outputs["first"][0], torch.tensor([[[1.11]]]))
    torch.testing.assert_close(outputs["second"][0], torch.tensor([[[2.22]]]))
    telemetry = runner.telemetry()
    assert telemetry["captures"] == 1
    assert telemetry["hits"] == 0
    assert telemetry["misses"] == 2
    assert telemetry["fallback_reasons"]["capture_in_progress"] == 1


def test_capture_cancellation_resets_graph_clears_marker_and_can_retry():
    runtime = _FakeRuntime()
    runner = _runner(runtime)
    runtime.fail_capture = _CaptureCancelledError()

    with pytest.raises(_CaptureCancelledError):
        _run(runner, 1.0)

    assert runner._capturing_keys == set()
    assert runtime.graphs[0].resets == 1
    assert runner.telemetry()["capture_failures"] == 0

    output = _run(runner, 2.0)
    torch.testing.assert_close(output[0], torch.tensor([[[2.22]]]))
    assert runner.telemetry()["captures"] == 1


def test_replay_failure_resets_entry_and_routes_future_calls_to_eager_only():
    runtime = _FakeRuntime()
    runner = _runner(runtime)
    _run(runner, 1.0)
    runtime.fail_replay = True

    second = _run(runner, 2.0)
    third = _run(runner, 3.0)

    torch.testing.assert_close(second[0], torch.tensor([[[2.22]]]))
    torch.testing.assert_close(third[0], torch.tensor([[[3.33]]]))
    telemetry = runner.telemetry()
    assert telemetry["cache_entries"] == 0
    assert telemetry["eager_only_keys"] == 1
    assert telemetry["fallback_reasons"]["replay_failure"] == 1
    assert telemetry["fallback_reasons"]["eager_only_key"] == 1
    assert runtime.graphs[0].resets == 1


def test_ineligible_fallbacks_and_telemetry_schema_are_fixed():
    runtime = _FakeRuntime()
    runner = _runner(runtime)

    runtime.supported = False
    _run(runner, 1.0)
    runtime.supported = True
    runtime.api_available = False
    _run(runner, 2.0)
    runtime.api_available = True
    runtime.capturing = True
    _run(runner, 3.0)

    telemetry = runner.telemetry()
    assert set(telemetry) == {
        "calls",
        "eligible_calls",
        "hits",
        "misses",
        "captures",
        "capture_failures",
        "eager_fallbacks",
        "budget_rejections",
        "evictions",
        "cache_entries",
        "cache_bytes",
        "eager_only_keys",
        "fallback_reasons",
    }
    assert telemetry["calls"] == 3
    assert telemetry["eligible_calls"] == 1
    assert telemetry["eager_fallbacks"] == 3
    assert telemetry["fallback_reasons"]["unsupported_device"] == 1
    assert telemetry["fallback_reasons"]["unsupported_api"] == 1
    assert telemetry["fallback_reasons"]["capture_in_progress"] == 1
    assert set(telemetry["fallback_reasons"]) == {
        "disabled",
        "unsupported_device",
        "unsupported_api",
        "capture_in_progress",
        "eager_only_key",
        "budget_rejection",
        "capture_failure",
        "replay_failure",
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_entries", True),
        ("max_entries", 0),
        ("max_entries", 1.0),
        ("max_bytes", False),
        ("max_bytes", -1),
        ("max_bytes", "1024"),
    ],
)
def test_runner_rejects_invalid_budgets(name, value):
    kwargs = {"max_entries": 4, "max_bytes": 1024, name: value}
    with pytest.raises(ValueError, match=name):
        NPUCFMGraphRunner(
            model_instance=object(),
            steps=10,
            enabled=True,
            runtime=_FakeRuntime(),
            **kwargs,
        )


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_runner_rejects_non_boolean_enabled(value):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        NPUCFMGraphRunner(
            model_instance=object(),
            steps=10,
            enabled=value,
            runtime=_FakeRuntime(),
        )
