# MiniCPM-o 4.5 low-risk optimizations

This document tracks opt-in, semantics-preserving experiments for the
MiniCPM-o 4.5 three-stage pipeline. All switches default to disabled. The
baseline API, sampling, dtype, CFM steps, and codec chunk sizes are unchanged.

## Tensor handoff

Set `VLLM_OMNI_MINICPMO45_TENSOR_HANDOFF=1` before starting the server to keep
the Thinker-to-Talker hidden-state slice as a contiguous FP32 tensor. The
Talker accepts both this representation and the legacy nested list.

This removes Python float object materialization. It is **not zero-copy**: the
stage request transport still transfers the tensor through host bytes. Tensor
values inside the runner-owned `model_intermediate_buffer` use a versioned
dtype/shape/bytes envelope so vLLM can recover their type from a field declared
as `Any`; ordinary lists, dictionaries, and scalars keep their representation.
The generic connector wire schema is unchanged.

## Deterministic initial-state template cache

Set `VLLM_OMNI_MINICPMO45_INITIAL_STATE_CACHE=1` to cache the deterministic
Code2Wav state produced for the configured official default reference voice.
`VLLM_OMNI_MINICPMO45_INITIAL_STATE_CACHE_MAX_ENTRIES` bounds the process-local
LRU and defaults to `1`. A non-integer or negative capacity is rejected; when
the cache is enabled the capacity must be at least one.

The key covers the backend instance and type, prompt ID, canonical path, WAV
content digest and sample rate, device, dtype, float16 mode, CFM step count,
encoder lookahead width, and template schema version. A cold miss runs the
unchanged setup path. A warm hit deep-clones every Flow and HiFT tensor for
each request, so the stored template is never passed to decode. Runtime-uploaded
or otherwise custom reference audio always runs the uncached setup path.

Set `VLLM_OMNI_MINICPMO45_PERF_STATS=1` to collect process-local host timing
and path counters, including initial-state hit/miss/eviction and setup/clone
host time. These counters do not synchronize the accelerator.

No Ascend A3 performance or full accuracy result is claimed by this change.
