from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
)
from vllm_omni.model_executor.models.minicpmo_4_5.optimization_config import (
    MINICPMO45_PERF_STATS,
    MiniCPMO45OptimizationConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.last_chunk_calls: list[bool] = []

    def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
        batch, length, _ = xs.shape
        self.calls.append(batch)
        self.last_chunk_calls.append(last_chunk)
        old_length = 0 if att_cache is None else att_cache.shape[3]
        output = xs[:, : max(1, length - 1)]
        cnn = xs[:, :1, :].transpose(1, 2).contiguous()
        marker = xs[:, 0, 0].reshape(1, batch, 1, 1, 1)
        att = marker.expand(1, batch, 1, old_length + output.shape[1], 1).clone()
        return output, cnn, att


class _FakeBlock:
    def __init__(self):
        conv1 = SimpleNamespace(causal_padding=(1, 0))
        self.conv = SimpleNamespace(
            in_channels=1,
            out_channels=1,
            block=[None, conv1],
        )
        self.attn = SimpleNamespace(num_heads=1, head_dim=1)


class _FakeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_FakeBlock()]
        self.cfg_batches: list[int] = []
        self.speaker_order: list[list[float]] = []

    def t_embedder(self, time):
        return time[:, None]

    def blocks_forward_chunk(
        self,
        inputs,
        time,
        mask,
        cnn_cache,
        att_cache,
        cnn_out,
        att_out,
    ):
        del time, mask, cnn_cache, att_cache
        self.cfg_batches.append(inputs.shape[0])
        self.speaker_order.append(inputs[:, 2, 0].tolist())
        marker = inputs[:, 1, 0]
        cnn_out.copy_(marker.reshape(1, -1, 1, 1).expand_as(cnn_out))
        att_out.copy_(marker.reshape(1, -1, 1, 1, 1).expand_as(att_out))
        return inputs[:, 1:2]


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _FakeEstimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.zeros(1, 1, 100), persistent=False)


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.encoder_proj = nn.Identity()
        self.decoder = _FakeDecoder()
        self.spk_embed_affine_layer = nn.Identity()

    def input_embedding(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1)


class _FakeHiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.sources: list[torch.Tensor] = []

    def forward(self, mel, source):
        self.calls.append(mel.shape[0])
        self.sources.append(source)
        speech = mel[:, 0].repeat_interleave(3, dim=1)
        generated_source = speech[:, None]
        return speech, generated_source


class _FakeToken2Wav:
    def __init__(self):
        self.flow = _FakeFlow()
        self.hift = _FakeHiFT()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.hamming_window(4, periodic=False)
        self.prompt_calls = 0

    def _prepare_prompt(self, prompt_wav):
        del prompt_wav
        self.prompt_calls += 1
        return (
            torch.tensor([[5, 6]], dtype=torch.long),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1, 1),
            torch.ones(1, 4, 1),
            torch.tensor([4], dtype=torch.int32),
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("sequential stream fallback must never be called")

    def __call__(self, *args, **kwargs):
        raise AssertionError("sequential __call__ fallback must never be called")


def _config(minimum: int = 1):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config={
                "extra": {
                    "code2wav_min_batch_size": minimum,
                    "prompt_cache_id": "shared",
                    "prompt_wav": "/fake/prompt.wav",
                }
            },
        )
    )


def _model(
    *,
    initial_state_cache: bool = False,
    initial_state_cache_max_entries: int = 1,
    batch1_low_copy: bool = False,
    perf_stats: bool = False,
):
    token2wav = _FakeToken2Wav()
    backend = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=initial_state_cache,
        initial_state_cache_max_entries=initial_state_cache_max_entries,
        batch1_low_copy=batch1_low_copy,
        perf_stats=perf_stats,
    )
    optimization_config = MiniCPMO45OptimizationConfig(
        initial_state_cache=initial_state_cache,
        initial_state_cache_max_entries=initial_state_cache_max_entries,
        batch1_low_copy=batch1_low_copy,
        perf_stats=perf_stats,
    )
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    model._optimization_config = optimization_config
    model.backend = backend
    return model, token2wav


def _write_prompt_wav(path: Path, *, sample_rate: int = 16000) -> None:
    import soundfile as sf

    sf.write(
        path,
        torch.linspace(-0.1, 0.1, 160).numpy(),
        sample_rate,
        format="WAV",
    )


def _state_digest(state) -> str:
    digest = sha256()
    for cache in (state.flow_cache, state.hift_cache):
        for name, value in sorted(cache.items()):
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_states_equal(actual, expected) -> None:
    assert actual.flow_cache.keys() == expected.flow_cache.keys()
    assert actual.hift_cache.keys() == expected.hift_cache.keys()
    for name in actual.flow_cache:
        torch.testing.assert_close(actual.flow_cache[name], expected.flow_cache[name])
    for name in actual.hift_cache:
        torch.testing.assert_close(actual.hift_cache[name], expected.hift_cache[name])


def _clone_states(states):
    return [BatchedToken2Wav._clone_state(state) for state in states]


def test_code2wav_resolves_hf_model_id_for_assets(mocker, tmp_path):
    resolved_root = tmp_path / "snapshot"
    resolved_root.mkdir()
    config = _config()
    config.model_config.model = "openbmb/MiniCPM-o-4_5"
    config.model_config.revision = "test-revision"
    config.model_config.stage_connector_config["extra"].pop("prompt_wav")
    config.load_config = SimpleNamespace(download_dir="/model-cache")
    model = MiniCPMO45Code2Wav(vllm_config=config)
    mock_download = mocker.patch(
        "vllm_omni.model_executor.model_loader.weight_utils.download_weights_from_hf_specific",
        return_value=str(resolved_root),
    )

    assert model._resolve_model_root() == resolved_root
    assert model.model_path == str(resolved_root)
    assert model._default_prompt_wav == str(resolved_root / "assets" / "HT_ref_audio.wav")
    mock_download.assert_called_once_with(
        "openbmb/MiniCPM-o-4_5",
        "/model-cache",
        allow_patterns=[
            "assets/HT_ref_audio.wav",
            "assets/token2wav/*",
        ],
        revision="test-revision",
        require_all=True,
    )


def _info(
    request_id: str,
    chunk_seq: int,
    codes: list[int],
    *,
    last_chunk: bool = False,
    cache_epoch: int = 0,
):
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long)},
        "meta": {
            "request_id": request_id,
            "chunk_seq": chunk_seq,
            "cache_epoch": cache_epoch,
            "last_chunk": last_chunk,
            "prompt_cache_id": "shared",
        },
    }


def _forward(model, infos, placeholder_counts=None, request_ids=None):
    placeholder_counts = placeholder_counts or [1] * len(infos)
    input_ids = torch.zeros(sum(placeholder_counts), dtype=torch.long)
    return model(
        input_ids=input_ids,
        seq_token_counts=placeholder_counts,
        runtime_additional_information=infos,
        request_ids=request_ids,
    )


def test_adapter_runs_true_batch_cfg_and_splits_request_caches():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.prompt_calls == 1
    assert token2wav.flow.encoder.calls == [2, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [4, 4, 4, 4]
    assert all(order == [1.0, 1.0, 0.0, 0.0] for order in token2wav.flow.decoder.estimator.speaker_order)
    assert token2wav.hift.calls == [2]
    assert len(audios) == 2
    cache0 = states[0].flow_cache["estimator_cnn_cache"]
    cache1 = states[1].flow_cache["estimator_cnn_cache"]
    assert cache0.data_ptr() != cache1.data_ptr()
    assert cache0[0, 0, 0, 0, 0].item() == 10
    assert cache1[0, 0, 0, 0, 0].item() == 20


def test_fade_in_out_limits_overlap_to_available_previous_audio():
    speech = torch.arange(6, dtype=torch.float32).reshape(1, -1)
    previous = torch.full((1, 3), 2.0)
    window = torch.hamming_window(8, periodic=False)

    actual = BatchedToken2Wav._fade_in_out(speech, previous, window)

    expected = speech.clone()
    expected[..., :3] = speech[..., :3] * window[:3] + previous * window[-3:]
    torch.testing.assert_close(actual, expected)


def test_estimator_cache_stack_split_round_trip_preserves_cfg_rows():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    stacked = adapter._stack_flow_cache(states)
    assert stacked["estimator_cnn_cache"].shape[2] == 4
    assert stacked["estimator_att_cache"].shape[2] == 4
    restored = adapter._split_flow_cache(stacked, 2)
    for original, round_tripped in zip(states, restored, strict=True):
        torch.testing.assert_close(
            round_tripped["estimator_cnn_cache"],
            original.flow_cache["estimator_cnn_cache"],
        )
        torch.testing.assert_close(
            round_tripped["estimator_att_cache"],
            original.flow_cache["estimator_att_cache"],
        )


def test_batch1_low_copy_matches_baseline_caches_and_waveforms():
    token2wav = _FakeToken2Wav()
    baseline = BatchedToken2Wav(token2wav, batch1_low_copy=False)
    candidate = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = baseline.prepare_prompt("shared", "/fake/prompt.wav")
    baseline_states = baseline.setup_batch(features, 1)
    candidate_states = _clone_states(baseline_states)

    for tokens, last_chunk in [([10, 11], False), ([12, 13], True)]:
        baseline_audio, baseline_states = baseline.decode_batch(
            torch.tensor([tokens]),
            features,
            baseline_states,
            last_chunk=last_chunk,
        )
        candidate_audio, candidate_states = candidate.decode_batch(
            torch.tensor([tokens]),
            features,
            candidate_states,
            last_chunk=last_chunk,
        )

        torch.testing.assert_close(candidate_audio[0], baseline_audio[0], rtol=0, atol=0)
        _assert_states_equal(candidate_states[0], baseline_states[0])


def test_batch1_low_copy_skips_singleton_flow_materialization():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)
    original = states[0]

    stacked = adapter._stack_flow_cache(states)
    assert stacked is original.flow_cache
    split = adapter._split_flow_cache(stacked, 1)
    for name, value in original.flow_cache.items():
        assert split[0][name].data_ptr() == value.data_ptr()
        assert split[0][name].untyped_storage().data_ptr() == value.untyped_storage().data_ptr()


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batch1_low_copy_keeps_batch_greater_than_one_on_baseline_path(
    batch_size,
    monkeypatch,
):
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, batch_size)
    baseline_stack = adapter._stack_flow_cache_baseline
    baseline_split = adapter._split_flow_cache_baseline
    stack_calls = 0
    split_calls = 0

    def tracked_stack(values):
        nonlocal stack_calls
        stack_calls += 1
        return baseline_stack(values)

    def tracked_split(cache, size):
        nonlocal split_calls
        split_calls += 1
        return baseline_split(cache, size)

    monkeypatch.setattr(adapter, "_stack_flow_cache_baseline", tracked_stack)
    monkeypatch.setattr(adapter, "_split_flow_cache_baseline", tracked_split)
    tokens = torch.arange(batch_size * 2).reshape(batch_size, 2) + 10

    adapter.decode_batch(tokens, features, states, last_chunk=False)

    assert stack_calls == 1
    # The tracker is installed after setup, so this is decode's split only.
    assert split_calls == 1


@pytest.mark.parametrize("batch_size", [2, 4])
def test_batch1_low_copy_matches_baseline_for_true_batches(batch_size):
    token2wav = _FakeToken2Wav()
    baseline = BatchedToken2Wav(token2wav, batch1_low_copy=False)
    candidate = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = baseline.prepare_prompt("shared", "/fake/prompt.wav")
    baseline_states = baseline.setup_batch(features, batch_size)
    candidate_states = _clone_states(baseline_states)

    chunks = [
        (torch.arange(batch_size * 2).reshape(batch_size, 2) + 10, False),
        (torch.arange(batch_size * 2).reshape(batch_size, 2) + 20, True),
    ]
    for tokens, last_chunk in chunks:
        baseline_audio, baseline_states = baseline.decode_batch(
            tokens,
            features,
            baseline_states,
            last_chunk=last_chunk,
        )
        candidate_audio, candidate_states = candidate.decode_batch(
            tokens,
            features,
            candidate_states,
            last_chunk=last_chunk,
        )

        for candidate_row, baseline_row in zip(candidate_audio, baseline_audio, strict=True):
            torch.testing.assert_close(candidate_row, baseline_row, rtol=0, atol=0)
        for candidate_state, baseline_state in zip(candidate_states, baseline_states, strict=True):
            _assert_states_equal(candidate_state, baseline_state)


def test_batch1_low_copy_disabled_keeps_singleton_on_baseline_path(monkeypatch):
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=False)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)
    baseline_stack = adapter._stack_flow_cache_baseline
    baseline_split = adapter._split_flow_cache_baseline
    stack_calls = 0
    split_calls = 0

    def tracked_stack(values):
        nonlocal stack_calls
        stack_calls += 1
        return baseline_stack(values)

    def tracked_split(cache, size):
        nonlocal split_calls
        split_calls += 1
        return baseline_split(cache, size)

    monkeypatch.setattr(adapter, "_stack_flow_cache_baseline", tracked_stack)
    monkeypatch.setattr(adapter, "_split_flow_cache_baseline", tracked_split)

    adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        states,
        last_chunk=False,
    )

    assert stack_calls == 1
    assert split_calls == 1


def test_batch1_low_copy_does_not_mutate_input_state():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)
    before = _clone_states(states)

    adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        states,
        last_chunk=False,
    )

    _assert_states_equal(states[0], before[0])


def test_batch1_low_copy_next_state_does_not_alias_input_storage():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)

    _, next_states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        states,
        last_chunk=False,
    )

    for old_cache, new_cache in (
        (states[0].flow_cache, next_states[0].flow_cache),
        (states[0].hift_cache, next_states[0].hift_cache),
    ):
        for name, old_value in old_cache.items():
            new_value = new_cache[name]
            if old_value.untyped_storage().nbytes() and new_value.untyped_storage().nbytes():
                assert old_value.untyped_storage().data_ptr() != new_value.untyped_storage().data_ptr()


def test_batch1_low_copy_hift_tails_use_compact_request_storage():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)

    _, next_states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        states,
        last_chunk=False,
    )

    for value in next_states[0].hift_cache.values():
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_batch1_low_copy_passes_request_owned_hift_source_directly():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, batch1_low_copy=True)
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        states,
        last_chunk=False,
    )
    old_source = states[0].hift_cache["source"]

    adapter.decode_batch(
        torch.tensor([[12, 13]]),
        features,
        states,
        last_chunk=True,
    )

    assert token2wav.hift.sources[-1] is old_source


def test_batch1_low_copy_perf_stats_count_skipped_materializations():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        batch1_low_copy=True,
        perf_stats=True,
    )
    features = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(features, 1)
    MINICPMO45_PERF_STATS.reset()

    adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        states,
        last_chunk=False,
    )

    stats = MINICPMO45_PERF_STATS.snapshot()
    assert stats["batch1_low_copy_count"] == 1
    assert stats["batch1_flow_stack_cat_skipped_count"] == 4
    assert stats["batch1_flow_split_cat_skipped_count"] == 2
    assert stats["batch1_flow_clone_skipped_count"] == 2
    assert stats["batch1_hift_stack_cat_skipped_count"] == 3
    # HiFT still needs three compact request-owned tail copies. The fast path
    # only moves where those copies occur, so it must not claim they vanished.
    assert stats["batch1_hift_clone_skipped_count"] == 0


def test_initial_state_cache_cold_miss_and_warm_hit_are_request_owned(tmp_path):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=1,
    )
    features = adapter.prepare_prompt("default", str(prompt_path))

    cold = adapter.setup_batch(
        features,
        1,
        prompt_cache_id="default",
        prompt_wav=str(prompt_path),
        allow_initial_state_cache=True,
    )
    setup_calls = list(token2wav.flow.encoder.calls)
    template = next(iter(adapter._initial_state_templates.values()))
    template_digest = _state_digest(template.state)
    warm = adapter.setup_batch(
        features,
        2,
        prompt_cache_id="default",
        prompt_wav=str(prompt_path),
        allow_initial_state_cache=True,
    )

    assert token2wav.flow.encoder.calls == setup_calls
    _assert_states_equal(warm[0], cold[0])
    _assert_states_equal(warm[1], cold[0])
    for cache_name in warm[0].flow_cache:
        assert warm[0].flow_cache[cache_name].data_ptr() != warm[1].flow_cache[cache_name].data_ptr()
        assert warm[0].flow_cache[cache_name].data_ptr() != template.state.flow_cache[cache_name].data_ptr()
    for cache_name in warm[0].hift_cache:
        assert warm[0].hift_cache[cache_name] is not warm[1].hift_cache[cache_name]
        assert warm[0].hift_cache[cache_name] is not template.state.hift_cache[cache_name]
        if warm[0].hift_cache[cache_name].numel() > 0:
            assert warm[0].hift_cache[cache_name].data_ptr() != warm[1].hift_cache[cache_name].data_ptr()
            assert warm[0].hift_cache[cache_name].data_ptr() != template.state.hift_cache[cache_name].data_ptr()

    warm[0].flow_cache["conformer_cnn_cache"].add_(1)
    assert _state_digest(template.state) == template_digest


def test_disabled_initial_state_cache_repeats_full_setup():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=False,
    )
    features = adapter.prepare_prompt("default", "/fake/default.wav")

    adapter.setup_batch(features, 1)
    adapter.setup_batch(features, 1)

    assert token2wav.flow.encoder.calls == [1, 1]
    assert adapter._initial_state_templates == {}


def test_initial_state_cache_hit_preserves_first_waveform(tmp_path):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=1,
    )
    features = adapter.prepare_prompt("default", str(prompt_path))
    cold = adapter.setup_batch(
        features,
        1,
        prompt_cache_id="default",
        prompt_wav=str(prompt_path),
        allow_initial_state_cache=True,
    )
    cold_audio, _ = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        cold,
        last_chunk=False,
    )
    warm = adapter.setup_batch(
        features,
        1,
        prompt_cache_id="default",
        prompt_wav=str(prompt_path),
        allow_initial_state_cache=True,
    )
    warm_audio, _ = adapter.decode_batch(
        torch.tensor([[10, 11]]),
        features,
        warm,
        last_chunk=False,
    )

    torch.testing.assert_close(warm_audio[0], cold_audio[0], rtol=0, atol=0)


def test_initial_state_cache_key_rejects_config_and_prompt_aliases(tmp_path):
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    _write_prompt_wav(first_path, sample_rate=16000)
    _write_prompt_wav(second_path, sample_rate=24000)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=8,
    )
    first_features = adapter.prepare_prompt("voice", str(first_path))

    first_key = adapter._initial_state_cache_key(
        first_features,
        prompt_cache_id="voice",
        prompt_wav=str(first_path),
    )
    other_id = adapter._initial_state_cache_key(
        first_features,
        prompt_cache_id="other",
        prompt_wav=str(first_path),
    )
    other_sample_rate = adapter._initial_state_cache_key(
        first_features,
        prompt_cache_id="voice",
        prompt_wav=str(second_path),
    )
    adapter.n_timesteps += 1
    other_steps = adapter._initial_state_cache_key(
        first_features,
        prompt_cache_id="voice",
        prompt_wav=str(first_path),
    )
    adapter.n_timesteps -= 1

    class _FakePreLookahead:
        pre_lookahead_len = 5

    token2wav.flow.encoder.pre_lookahead_layer = _FakePreLookahead()
    other_lookahead = adapter._initial_state_cache_key(
        first_features,
        prompt_cache_id="voice",
        prompt_wav=str(first_path),
    )

    assert len({first_key, other_id, other_sample_rate, other_steps, other_lookahead}) == 5
    assert first_key.prompt_sample_rate == 16000
    assert other_sample_rate.prompt_sample_rate == 24000


def test_initial_state_cache_digest_detects_in_place_prompt_change(tmp_path):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=2,
    )
    features = adapter.prepare_prompt("default", str(prompt_path))
    first_key = adapter._initial_state_cache_key(
        features,
        prompt_cache_id="default",
        prompt_wav=str(prompt_path),
    )
    prompt_path.write_bytes(prompt_path.read_bytes() + b"changed")
    second_key = adapter._initial_state_cache_key(
        features,
        prompt_cache_id="default",
        prompt_wav=str(prompt_path),
    )

    assert first_key.prompt_sha256 != second_key.prompt_sha256


def test_initial_state_cache_lru_and_prompt_evict(tmp_path):
    paths = [tmp_path / f"prompt-{index}.wav" for index in range(3)]
    for index, path in enumerate(paths):
        _write_prompt_wav(path, sample_rate=16000 + index)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=2,
    )

    for index, path in enumerate(paths):
        features = adapter.prepare_prompt(f"voice-{index}", str(path))
        adapter.setup_batch(
            features,
            1,
            prompt_cache_id=f"voice-{index}",
            prompt_wav=str(path),
            allow_initial_state_cache=True,
        )

    keys = list(adapter._initial_state_templates)
    assert [key.prompt_cache_id for key in keys] == ["voice-1", "voice-2"]
    adapter.evict_prompt("voice-1", str(paths[1]))
    assert [key.prompt_cache_id for key in adapter._initial_state_templates] == ["voice-2"]


def test_initial_state_cache_failure_never_publishes_template(tmp_path, monkeypatch):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=1,
    )
    features = adapter.prepare_prompt("default", str(prompt_path))

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected setup failure")

    monkeypatch.setattr(adapter, "_setup_batch_uncached", fail)
    with pytest.raises(RuntimeError, match="injected setup failure"):
        adapter.setup_batch(
            features,
            1,
            prompt_cache_id="default",
            prompt_wav=str(prompt_path),
            allow_initial_state_cache=True,
        )

    assert adapter._initial_state_templates == {}


def test_initial_state_cache_perf_stats_are_host_only(tmp_path):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(
        token2wav,
        initial_state_cache_enabled=True,
        initial_state_cache_max_entries=1,
        perf_stats=True,
    )
    features = adapter.prepare_prompt("default", str(prompt_path))
    MINICPMO45_PERF_STATS.reset()

    for _ in range(2):
        adapter.setup_batch(
            features,
            1,
            prompt_cache_id="default",
            prompt_wav=str(prompt_path),
            allow_initial_state_cache=True,
        )

    stats = MINICPMO45_PERF_STATS.snapshot()
    assert stats["initial_state_cache_miss_count"] == 1
    assert stats["initial_state_cache_hit_count"] == 1
    assert stats["initial_state_setup_ns"] >= 0
    assert stats["initial_state_clone_ns"] >= 0


def test_model_preserves_output_slots_and_prefers_runtime_codes():
    model, token2wav = _model()
    output = _forward(
        model,
        [_info("a", 0, [10, 11]), _info("b", 0, [20, 21])],
        placeholder_counts=[3, 1],
    )

    audios = output.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert len(output.multimodal_outputs["sr"]) == 2
    assert all(sr.item() == 24000 for sr in output.multimodal_outputs["sr"])
    assert all(audio.dtype == torch.float32 for audio in audios)
    # Fake CFM uses two Euler steps whose deltas sum to one. Its conditional
    # row is mu and its unconditional row is zero, so CFG produces 1.7 * mu.
    torch.testing.assert_close(audios[0][0], torch.tensor(1.7 * 10))
    torch.testing.assert_close(audios[1][0], torch.tensor(1.7 * 20))
    assert token2wav.flow.encoder.calls[-1] == 2


def test_code2wav_projects_duplex_metadata_to_final_audio_output():
    model, token2wav = _model()
    segment = _info("duplex", 0, [10, 11])
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    segment["meta"].update(
        {
            "duplex_epoch": 3,
            "duplex_turn_id": 7,
            "llm_output_text_utf8": segment_text_utf8,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    segment_output = _forward(model, [segment])

    assert segment_output.multimodal_outputs["meta.turn_end"][0].item() is False
    # A Talker unit boundary only drains pending codec tokens. The official
    # streaming path keeps Token2wav open until the assistant turn ends.
    assert token2wav.flow.encoder.last_chunk_calls[-1] is False
    assert "duplex" in model._states

    final = _info("duplex", 1, [12, 13], last_chunk=True)
    final["meta"].update(segment["meta"])
    final["meta"]["chunk_seq"] = 1
    final["meta"]["last_chunk"] = True
    final["meta"]["turn_end"] = True
    output = _forward(model, [final])

    payload = output.multimodal_outputs
    assert "meta" not in payload
    assert payload["meta.duplex_epoch"][0].item() == 3
    assert payload["meta.duplex_turn_id"][0].item() == 7
    torch.testing.assert_close(
        payload["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert payload["meta.tts_is_last_chunk"][0].item() is True
    assert payload["meta.turn_end"][0].item() is True
    assert token2wav.flow.encoder.last_chunk_calls[-1] is True
    assert "duplex" not in model._states


def test_initial_empty_segment_marker_initializes_stream_without_audio():
    model, token2wav = _model()
    boundary = _info("duplex", 0, [])
    boundary["meta"].update(
        {
            "code_flat_numel": 0,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    output = _forward(model, [boundary])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert "duplex" in model._states
    assert token2wav.hift.calls == []

    resumed = _info(
        "duplex",
        1,
        [4218, 4218, 4218, 10, 11, 12, 13, 14],
    )
    output = _forward(model, [resumed])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert "duplex" in model._states


def test_code2wav_cache_only_qualifies_official_default_prompt(tmp_path):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    model, _ = _model(initial_state_cache=True)
    model._default_prompt_id = "default"
    model._default_prompt_wav = str(prompt_path)

    default = _info("default-request", 0, [10, 11])
    default["meta"]["prompt_cache_id"] = "default"
    default["meta"]["prompt_wav"] = str(prompt_path)
    _forward(model, [default])
    assert len(model.backend._initial_state_templates) == 1

    custom_path = tmp_path / "custom.wav"
    _write_prompt_wav(custom_path)
    custom = _info("custom-request", 0, [12, 13])
    custom["meta"]["prompt_cache_id"] = "custom"
    custom["meta"]["prompt_wav"] = str(custom_path)
    _forward(model, [custom])
    assert len(model.backend._initial_state_templates) == 1

    runtime = _info("runtime-request", 0, [14, 15])
    runtime["codes"]["ref"] = torch.tensor([0.0, 0.25, -0.25, 0.0])
    runtime["meta"]["ref_audio_sr"] = 16000
    runtime["meta"].pop("prompt_cache_id")
    _forward(model, [runtime])
    assert len(model.backend._initial_state_templates) == 1


def test_initial_state_cache_soak_keeps_template_and_request_maps_bounded(
    tmp_path,
):
    prompt_path = tmp_path / "default.wav"
    _write_prompt_wav(prompt_path)
    model, _ = _model(initial_state_cache=True)
    model._default_prompt_id = "default"
    model._default_prompt_wav = str(prompt_path)

    for index in range(100):
        info = _info(f"request-{index}", 0, [10, 11], last_chunk=True)
        info["meta"]["prompt_cache_id"] = "default"
        info["meta"]["prompt_wav"] = str(prompt_path)
        _forward(model, [info])
        model.on_requests_finished([f"request-{index}"])

    assert len(model.backend._initial_state_templates) == 1
    assert model._states == {}
    assert model._request_prompt_keys == {}
    assert model._runtime_prompts == {}


def test_shared_runtime_prompt_recreates_missing_file_before_second_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    first = _info("voice-a", 0, [10, 11])
    first["codes"]["ref"] = reference
    first["meta"]["ref_audio_sr"] = 16000
    first["meta"].pop("prompt_cache_id")
    _forward(model, [first], request_ids=["internal-a"])

    prompt_key = model._request_prompt_keys["internal-a"]
    prompt_path = Path(model._runtime_prompts[prompt_key].path)
    prompt_path.unlink()

    second = _info("voice-b", 0, [12, 13])
    second["codes"]["ref"] = reference
    second["meta"]["ref_audio_sr"] = 16000
    second["meta"].pop("prompt_cache_id")
    _forward(model, [second], request_ids=["internal-b"])

    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-a", "internal-b"}

    model.on_requests_finished(["internal-a"])
    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-b"}

    model.on_requests_finished(["internal-b"])
    assert not prompt_path.exists()
    assert prompt_key not in model._runtime_prompts


def test_runtime_prompt_write_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.sf.write",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated write failure"):
        model._materialize_runtime_prompt(reference, 16000)

    assert len(model._runtime_prompts) == 1
    entry = next(iter(model._runtime_prompts.values()))
    assert not Path(entry.path).exists()
    assert list(Path(entry.path).parent.iterdir()) == []


def test_runtime_prompt_files_are_isolated_between_model_instances(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    first_model, _ = _model()
    second_model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def runtime_ref_info(request_id: str):
        info = _info(request_id, 0, [10, 11])
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    _forward(first_model, [runtime_ref_info("voice-a")], request_ids=["internal-a"])
    _forward(second_model, [runtime_ref_info("voice-b")], request_ids=["internal-b"])

    first_key = first_model._request_prompt_keys["internal-a"]
    second_key = second_model._request_prompt_keys["internal-b"]
    first_path = Path(first_model._runtime_prompts[first_key].path)
    second_path = Path(second_model._runtime_prompts[second_key].path)
    assert first_key == second_key
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()

    first_model.on_requests_finished(["internal-a"])
    assert not first_path.exists()
    assert second_path.is_file()

    second_model.on_requests_finished(["internal-b"])
    assert not second_path.exists()


def test_mixed_final_exact_buckets_keep_order_and_release_only_final_states():
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    output = _forward(
        model,
        [
            _info("a", 1, [11, 12]),
            _info("c", 1, [31, 32, 33], last_chunk=True),
            _info("b", 1, [21, 22]),
            _info("d", 1, [41, 42, 43], last_chunk=True),
        ],
    )

    audios = output.multimodal_outputs["model_outputs"]
    window = torch.hamming_window(4, periodic=False)
    overlap_scale = 1.7 * (window[0] + window[2])
    expected = torch.tensor([1, 3, 2, 4], dtype=torch.float32) * overlap_scale
    actual = torch.stack([audio[0] for audio in audios])
    torch.testing.assert_close(actual, expected)
    assert set(model._states) == {"a", "b"}


def test_empty_final_sentinel_emits_empty_and_releases_state_without_compute():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    hift_calls = list(token2wav.hift.calls)
    output = _forward(
        model,
        [
            _info("a", 1, [], last_chunk=True),
            _info("b", 1, [], last_chunk=True),
        ],
    )

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}
    assert token2wav.hift.calls == hift_calls


def test_empty_final_ignores_generation_scheduler_placeholder_token():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    infos = [_info("a", 1, [], last_chunk=True), _info("b", 1, [], last_chunk=True)]
    for info in infos:
        info.pop("codes")
        info["meta"]["code_flat_numel"] = 0

    output = _forward(model, infos, placeholder_counts=[1, 1])

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}


@pytest.mark.parametrize(
    "info",
    [
        # The runner injects the engine request id on every step (GPU
        # _preprocess, NPU _gather_runtime_additional_information)...
        {"request_id": "a", "meta": {"request_id": "a"}},
        # ...but a pre-warm step can also reach the model with nothing at all.
        {},
    ],
)
def test_prewarm_placeholder_step_emits_silence_without_touching_state(info):
    # async-chunk pre-warm submits Stage 2 with a reserved placeholder prompt.
    # If it gets scheduled before the first codec window lands, those reserved
    # tokens must neither be vocoded nor held to the codec payload contract.
    model, token2wav = _model()

    output = _forward(model, [info], request_ids=["a"])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._states == {}
    assert token2wav.hift.calls == []


def test_metadata_only_payload_still_decodes_codec_from_prompt_tokens():
    # The connector strips 1-D codec tensors out of additional_information and
    # leaves them in the prompt tokens, so a real chunk reaches the model as
    # producer metadata plus input ids. It must still be vocoded.
    model, _ = _model()
    info = {
        "request_id": "a",
        "meta": {
            "request_id": "a",
            "chunk_seq": 0,
            "code_flat_numel": 2,
            "prompt_cache_id": "shared",
        },
    }

    output = _forward(model, [info], placeholder_counts=[2])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert set(model._states) == {"a"}


def test_non_final_chunk_shorter_than_lookahead_window_is_rejected():
    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    with pytest.raises(RuntimeError, match="chunk_below_lookahead_window"):
        adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=False)

    # The final chunk is zero-padded by the encoder, so it stays decodable.
    audios, _ = adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=True)
    assert len(audios) == 1


def test_forward_builds_backend_when_weight_loading_was_skipped(monkeypatch):
    # load_format=dummy never calls load_weights(), so Stage 2 would otherwise
    # reach its first request with no Token2wav assets at all.
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    token2wav = _FakeToken2Wav()
    builds = 0

    def build_backend():
        nonlocal builds
        builds += 1
        model.backend = BatchedToken2Wav(token2wav)

    monkeypatch.setattr(model, "_build_backend", build_backend)

    output = _forward(model, [_info("a", 0, [10, 11])])
    _forward(model, [_info("a", 1, [12, 13])])

    assert builds == 1
    assert output.multimodal_outputs["model_outputs"][0].numel() > 0


@pytest.mark.parametrize(
    ("info", "reason"),
    [
        (_info("a", 0, [1, 2], cache_epoch=-1), "negative_stream_position"),
        (_info("a", 0, [1, 2]), "stale_or_reordered_chunk"),
        (_info("a", 2, [1, 2]), "stale_or_reordered_chunk"),
    ],
)
def test_stale_epoch_and_reordered_chunks_are_rejected(info, reason):
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])

    with pytest.raises(RuntimeError, match=reason):
        _forward(model, [info, _info("b", 1, [3, 4])])


def test_singleton_and_mixed_shape_buckets_use_same_batched_backend_without_fallback():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    output = _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8, 9])])

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    # Exact-shape buckets execute independently but both use the same vectorized
    # adapter; there is no Token2wav.stream/__call__ fallback.
    assert token2wav.hift.calls[-2:] == [1, 1]


def test_backend_failure_does_not_commit_any_request_state(monkeypatch):
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    before = dict(model._states)
    original = model.backend.decode_batch
    call_count = 0

    def fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "decode_batch", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _forward(
            model,
            [
                _info("a", 1, [5, 6]),
                _info("b", 1, [7, 8]),
                _info("c", 1, [9, 10, 11]),
                _info("d", 1, [12, 13, 14]),
            ],
        )
    assert call_count == 2
    assert model._states == before


def test_cleanup_and_profile_output_are_aligned():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    model.on_requests_finished(["a"])
    assert set(model._states) == {"b"}

    profile = model(
        input_ids=torch.zeros(5, dtype=torch.long),
        seq_token_counts=[2, 3],
    )
    assert [audio.numel() for audio in profile.multimodal_outputs["model_outputs"]] == [0, 0]
    assert set(model._states) == {"b"}


def test_cleanup_uses_generation_runner_internal_request_ids():
    model, _ = _model()
    _forward(
        model,
        [_info("external-a", 0, [1, 2]), _info("external-b", 0, [3, 4])],
        request_ids=["internal-a", "internal-b"],
    )

    model.on_requests_finished(["internal-a"])

    assert set(model._states) == {"internal-b"}


def test_reference_voice_and_duplex_metadata_follow_request_lifecycle():
    model, _ = _model()
    first = _info("voice-a", 0, [1, 2])
    first["codes"]["ref"] = torch.linspace(-0.1, 0.1, 160)
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    first["meta"].update(
        ref_audio_sr=16000,
        llm_output_text_utf8=segment_text_utf8,
        duplex_turn_id=7,
        duplex_epoch=3,
    )
    first["meta"].pop("prompt_cache_id")

    output = _forward(model, [first])
    prompt_key = model._request_prompt_keys["voice-a"]
    prompt = model._runtime_prompts[prompt_key]
    prompt_cache_id, prompt_wav = prompt.cache_id, prompt.path
    assert prompt_cache_id.startswith("runtime-ref-")
    assert Path(prompt_wav).is_file()
    torch.testing.assert_close(
        output.multimodal_outputs["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert output.multimodal_outputs["meta.duplex_turn_id"][0].item() == 7
    assert output.multimodal_outputs["meta.duplex_epoch"][0].item() == 3

    final = _info("voice-a", 1, [3, 4], last_chunk=True)
    final["meta"].pop("prompt_cache_id")
    final["meta"]["tts_is_last_chunk"] = True
    output = _forward(model, [final])

    assert output.multimodal_outputs["meta.tts_is_last_chunk"][0].item() is True
    assert model._request_prompt_keys["voice-a"] == prompt_key
    model.on_requests_finished(["voice-a"])
    assert "voice-a" not in model._request_prompt_keys
    assert prompt_key not in model._runtime_prompts
    assert not Path(prompt_wav).exists()
    assert (prompt_cache_id, prompt_wav) not in model.backend._prompt_features
