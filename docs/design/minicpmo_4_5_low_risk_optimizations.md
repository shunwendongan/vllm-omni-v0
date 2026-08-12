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

Set `VLLM_OMNI_MINICPMO45_PERF_STATS=1` to collect process-local host timing
and path counters. These counters do not synchronize the accelerator.

No Ascend A3 performance or full accuracy result is claimed by this change.
