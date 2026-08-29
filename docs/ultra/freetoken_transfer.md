# FreeToken ideas transferred to MiniCPM-o 4.5

Status: implemented behind default-off switches; local/static validation only.
No Atlas A3/910C performance or full-quality claim is made here.

## Source and boundary

This audit pins FreeToken `main` at
[`58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6`](https://github.com/FlashML-org/FreeToken/tree/58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6)
and the FreeToken paper at arXiv `2608.16157v1`. FreeToken is an edge MoE engine
for heterogeneous CPU/GPU machines; MiniCPM-o 4.5 in this challenge is a
single-NPU, dense, three-stage multimodal/TTS pipeline. FreeToken results are
therefore hypotheses and design evidence, not portable speedup claims.

The Ultra baseline already has Tensor handoff, batch-one low-copy Code2Wav,
early Code2Wav setup, Talker batch sampling, first/steady chunk separation,
bounded Estimator NPUGraphs, and isolated step/precision switches. This work
does not relabel those capabilities as FreeToken-derived gains.

## Transfer map

| FreeToken mechanism | Source anchor | MiniCPM-o transfer | Expected local effect | Status |
| --- | --- | --- | --- | --- |
| Stable graph buffers and replay-time input copy | [`engine/graph.py`](https://github.com/FlashML-org/FreeToken/blob/58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6/python/freetoken/engine/graph.py) | Exact-signature whole CFM-loop NPUGraph with persistent timeline/time-embedding/Euler tensors | Remove Python dispatch and repeated per-step allocation/copy from the 10/8/6-step loop | implemented, default off |
| Graph-resident dynamic work expressed as fixed-shape buffers and valid state | [paper §4.1](https://arxiv.org/html/2608.16157#S4.SS1) | Cache key includes bucket, cached state, steps, precision, shape, dtype, and device; capacity is bounded | Preserve address stability while preventing unbounded graph/HBM growth | implemented, default off |
| Full-layer double buffering on a dedicated transfer stream | [`moe/offload_cache.py`](https://github.com/FlashML-org/FreeToken/blob/58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6/python/freetoken/moe/offload_cache.py) | Request-owned two-slot audio staging ring plus one bounded encoder worker | Overlap a completed chunk's D2H/encoding with subsequent Stage 2 work | implemented, default off |
| Explicit cache/graph/activation memory budget | [`engine/cache_budget.py`](https://github.com/FlashML-org/FreeToken/blob/58f4b9ec0e166205c4dfd0c6ec184ea83b5957e6/python/freetoken/engine/cache_budget.py) | Maximum eight whole-loop graphs, exact buckets, observable workspace bytes, fallback on capacity/shape miss | Bound graph residency and keep the baseline recoverable | implemented, default off |
| Measured, restart-isolated policies rather than assumed portability | [paper §3.2](https://arxiv.org/html/2608.16157#S3.SS2) | Six fixed numerical arms and an official-runner evidence matrix | Separate 10/8/6-step and FP32/FP16 effects under identical request order | driver locally validated; A3 pending |

## Whole-loop CFM NPUGraph

`freetoken/cfm-loop-graph` adds a batch-one exact-signature graph above the
existing one-step Estimator graph. One replay covers the cosine schedule and
all Euler/CFG iterations. The graph owns fixed-address dynamic-input staging,
time embeddings, `dt`, CFG zero tensors, cache workspaces, and outputs. Request
code copies only dynamic inputs into these buffers and clones request-owned
final state once.

The key distinguishes setup/first/steady/tail buckets, cached versus uncached,
10/8/6 steps, effective Flow precision, every input shape/dtype, and device.
`code2wav_enable_cfm_loop_npu_graph` or
`VLLM_OMNI_MINICPMO45_CFM_LOOP_NPU_GRAPH` opts in;
`VLLM_OMNI_MINICPMO45_CFM_LOOP_MAX_GRAPHS` defaults to eight. Batch greater
than one, non-NPU, unsupported APIs, a capacity miss, or an unknown signature
uses the original CFM function, which can still select the Estimator Graph and
then eager. Once capture begins, capture failure is fatal because allocator or
graph state can no longer be assumed reusable.

Default-off timeline fields report capture/replay/miss, exact bucket, graph
capacity, workspace bytes, and fallback reason without synchronizing the NPU.

## Audio output pipeline

`freetoken/audio-output-pipeline` introduces a request/epoch-owned two-slot
ring. A slot records sequence and generation so it cannot be recycled while an
older encoding ticket still owns it. NPU tensors use a dedicated copy stream,
pinned host storage, and a local event; no NPU event, stream, or device handle
crosses the SharedMemoryConnector. CPU tensors use the same ordering state
without pretending the copy is asynchronous.

A single-worker executor keeps WAV/PCM/Base64 encoding bounded and ordered.
Abort, exception, epoch replacement, final flush, and normal completion all
fence or release request state. Empty control audio bypasses the fast path and
cannot become the first valid audio packet. Unsupported device, pinned-memory
failure, buffer mismatch, or a disabled
`VLLM_OMNI_MINICPMO45_ASYNC_AUDIO_OUTPUT` flag uses the legacy synchronous
path.

## Ideas deliberately not transferred

- FreeToken's shared expert LRU and CPU/GPU `q*` split solve MoE weight misses.
  MiniCPM-o Code2Wav is dense and resident on one NPU, so there is no equivalent
  expert working set or host-expert execution path.
- FTW changes checkpoint storage and startup. Challenge startup is outside the
  primary RTF/TTFP/TTFT score and the submitted model weights must remain
  compatible, so the format is not introduced.
- Semantic anchors target edited agent histories and recurrent/KV state. The
  Seed-TTS score path is not an edited multi-turn agent trace.
- Runtime cache resizing requires scheduler safe points and rebuild evidence.
  This iteration uses finite graph/ring capacities instead of adding a general
  memory manager.

## Repeated audio/video input proposal

Removing repeated audio samples, spectrogram frames, or video frames is not a
value-equivalent optimization. It changes timestamps, positional embeddings,
motion/audio duration, speaker/prosody evidence, and possibly the target answer;
it must not alter official Seed-TTS, Daily-Omni, or Video-MME inputs.

The safe version is exact-content reuse: hash the complete decoded media plus
sample rate/layout and cache immutable preprocessing features under a bounded
LRU. Ultra already hashes runtime TTS reference waveforms and caches Code2Wav
prompt features while request owners exist. The official Seed-TTS loader uses
fixed, non-oversampled rows whose reference WAV normally changes per request;
only endpoint checks/warmups repeat the first sample. A longer-lived cache is
therefore a future experiment, not part of this integration.

It should be implemented only after a timeline shows meaningful Stage 0 or
prompt-extraction time and a trace shows reusable content. Required gates are:
exact byte/format keying, model/revision/preprocessor parameters in the key,
immutable cached tensors, per-request ownership on read, bounded bytes and
eviction, hit/miss/bytes telemetry, abort/epoch isolation, and unchanged model
inputs. Promotion requires a meaningful official-run hit rate, lower TTFT or
TTFP under the frozen request order, no RTF/memory regression, and complete
Daily-Omni/Video-MME/ASV/WER parity.

## Evidence and promotion

The numerical driver is `scripts/run_minicpmo45_numerical_matrix.py`; its
official-runner argv example is
`docs/ultra/minicpmo45_numerical_matrix.example.json`. It records a new evidence
directory, commit/branch/dirty state, host details, official-config SHA256,
rendered argv, arm environment, raw result JSON, logs, and summaries. Each arm
gets a clean server process, two official warmups, and `B-C-C-B-B-C`; profiler
runs must remain outside scored evidence.

Promotion order is RTF, then TTFP, then TTFT. Require either at least 2% RTF
improvement with TTFP/TTFT regression no worse than 1%, or RTF non-inferiority
within 0.5% with at least 5% TTFP improvement. Peak memory growth is limited to
5%; requests, decodable audio, and continuity must remain 100%. Quality gates
are Daily-Omni >= 77.5%, Video-MME >= 67%, ASV >= 0.689, and WER <= 1.56%.
Until those A3 and complete-quality runs exist, the highest valid status is
local validation.
