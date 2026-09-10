# MiniCPM-o 4.5 Ascend NPU Inference Optimization

This branch is a focused vLLM-Omni optimization stack for MiniCPM-o 4.5 on
Ascend NPU. It keeps the original Thinker-Talker-Code2Wav serving topology and
targets the runtime hot path that decides TTFT, TTFP, E2E latency, and audio
RTF.

The implementation is based on vLLM-Omni and specializes the MiniCPM-o 4.5
full-modality pipeline for an Ascend 910C / Atlas A3 style environment. The
main work is in graph execution, autoregressive scheduling, cross-stage tensor
handoff, and streaming audio generation.

## Branch Scope

| Item | Value |
|---|---|
| Branch | `vllm-omni-v0-max_others` |
| Optimization evidence commit | `ea66d2d2f3aba0adc021046860423273dde249d7` |
| Target model | MiniCPM-o 4.5 |
| Main deploy config | `vllm_omni/deploy/minicpmo_4_5.yaml` |
| Target pipeline | Stage 0 Thinker -> Stage 1 Talker -> Stage 2 Code2Wav |
| Target backend | Ascend NPU through torch-npu / vLLM-Ascend / CANN |

This README describes the optimization branch, not the upstream vLLM-Omni
project in general. Upstream documentation remains the best reference for the
base framework APIs and supported model families.

## Performance Summary

The final-stack numbers below are historical same-metric results kept in the
branch history. They are cumulative results from multiple optimizations and
must not be attributed to one single change.

| Seed-TTS concurrency | Official baseline RTF | Optimized RTF | RTF reduction |
|---:|---:|---:|---:|
| 1 | 0.4423 | 0.3047 | 31.1% |
| 4 | 1.5734 | 0.4427 | 71.9% |
| 8 | 2.3024 | 0.6298 | 72.6% |

For concurrency 1, first-response latency improved from:

| Metric | Official baseline | Optimized | Reduction |
|---|---:|---:|---:|
| TTFT | 333.27 ms | 257.8 ms | 22.6% |
| TTFP | 986.47 ms | 364.0 ms | 63.1% |

Quality and guardrail records in the branch history include:

- K14 n-gram speculative decoding and K12 runner-local decode: WER and SIM
  unchanged in the recorded A/B runs.
- H1 EOS early termination: downstream audio remained bitwise identical while
  Stage 0 output tokens dropped from 435 to 403 in the recorded run.
- TJS1 pseudo-1step CFM: full zh2020 gate passed with WER 1.03% and SIM 0.8381
  against the recorded gates WER <= 1.56% and SIM >= 0.689.
- Historical full-task checks recorded Daily-Omni 78.09% and Video-MME 69.59%,
  both above the corresponding admission thresholds in the preserved report.

## Optimization Stack

### 1. Decode Graph Defaults

Auto-regressive stages are forced to `FULL_DECODE_ONLY` graph mode with
stage-specific capture buckets. Stage 0 uses larger buckets for Thinker decode,
while Stage 1 uses smaller Talker buckets. The same configuration path also
enables Ascend static-kernel support when the backend provides it.

Relevant code:

- `vllm_omni/config/stage_config.py`
- `vllm_omni/platforms/npu/graph_tools.py`

### 2. Stage 0 K14 n-gram Speculative Decoding

Stage 0 defaults to n-gram speculative decoding when the official deploy config
does not provide its own `speculative_config`. The current default reads
`OMNI_TALKER_S0SPEC_K`, falling back to `14`.

The recorded A/B compared K10 and K14:

| Change | Recorded effect |
|---|---|
| K10 -> K14 | E2E latency -10.0 ms, RTF -0.0027 |
| K14 guardrail | WER/SIM unchanged |
| K15 result | No-go due to extra verification cost and WER regression |

This is prompt/context lookup based speculation. It does not use a separate
draft model and it does not mean 14 final tokens are generated in parallel.

### 3. H1 EOS Early Termination

Stage 0 injects MiniCPM-o specific stop tokens for the TTS path:

- `151704`: TTS EOS
- `151645`: IM END

The Talker/Code2Wav path slices the TTS segment by BOS/EOS, so decoding beyond
that boundary is downstream-unused tail work. Stopping there reduced the
recorded E2E latency by 31.5 ms and kept downstream audio bitwise identical.

Rollback:

```bash
export OMNI_TALKER_H1_STOP=0
```

### 4. Stage 1 K12 Runner-local Multi-step Decode

The Talker stage runs multiple sequential one-token decode steps inside one
runner round trip when the batch is eligible. This does not make the
autoregressive dependency parallel. It amortizes Scheduler -> IPC -> Runner ->
output -> Scheduler overhead across a local K-token window.

The branch default is K12 in both the scheduler accounting path and the runner
path. The recorded sandwich A/B for K8 -> K12 showed 12.5 ms E2E latency
reduction with WER/SIM unchanged.

Useful knobs:

```bash
export OMNI_TALKER_SCHED_K=12
export OMNI_TALKER_LOCAL_STEPS=12
export OMNI_TALKER_LOCAL_DECODE=1
```

Disable local decode:

```bash
export OMNI_TALKER_LOCAL_DECODE=0
```

### 5. Tagged Raw-bytes Stage Handoff

The Stage 0 -> Stage 1 tensor handoff replaces nested Python list serialization
with a tagged raw-bytes payload carrying dtype, shape, and contiguous tensor
bytes.

Old path:

```text
Tensor -> CPU -> nested Python list -> msgpack
```

Optimized path:

```text
Tensor -> CPU contiguous bytes + dtype + shape -> receiver reconstructs Tensor
```

The recorded A/B showed E2E latency -14.1 ms and RTF -0.0032. This is not
zero-copy: the sender still materializes CPU bytes and the receiver still
reconstructs a tensor.

Rollback:

```bash
export VLLM_OMNI_HANDOFF_LIST_LEGACY=1
```

### 6. Code2Wav Exact-signature NPU Graph

Stage 2 defaults enable bounded NPU Graph acceleration for Code2Wav and HiFT:

```text
code2wav_enable_npu_graph = true
enable_hift_npu_graph = true
code2wav_max_npu_graphs = 48
hift_npu_graph_max_graphs = 8
```

The graph runner keys captures by operation constants plus input shape, dtype,
and device. New signatures run eager when the graph budget is exhausted. A
capture failure is treated as a stage-level failure instead of a silent
fallback, because graph capture state can poison the worker.

### 7. TJS1 pseudo-1step CFM

`OMNI_TJS_STOP=1` activates the pseudo-1step CFM path for Code2Wav. This is an
algorithmic approximation rather than a bitwise-equivalent runtime optimization,
so it is only valid together with quality gates.

Recorded result:

| Change | Recorded effect |
|---|---|
| pseudo-2step -> pseudo-1step | E2E latency -12 ms |
| zh2020 quality gate | WER 1.03%, SIM 0.8381 |

Set the mode explicitly:

```bash
export OMNI_TJS_STOP=1
```

## Code Map

| Area | Files |
|---|---|
| Stage config, Graph defaults, H1, K14, Stage 2 graph toggles | `vllm_omni/config/stage_config.py` |
| Runner-local K-window scheduling | `vllm_omni/core/sched/omni_ar_scheduler.py` |
| NPU Talker runner local decode | `vllm_omni/platforms/npu/worker/npu_ar_model_runner.py` |
| Tagged raw-bytes sender | `vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py` |
| Tagged raw-bytes receiver | `vllm_omni/experimental/fullduplex/engine/intermediate.py` |
| Exact-signature NPU Graph cache | `vllm_omni/platforms/npu/graph_tools.py` |
| Code2Wav / CFM path | `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py` |

## Run

Install the project with the same Python and backend stack required by
vLLM-Omni, vLLM-Ascend, torch-npu, and CANN for the target machine.

```bash
git clone -b vllm-omni-v0-max_others https://github.com/shunwendongan/vllm-omni-v0.git
cd vllm-omni-v0
pip install -e .
```

Start MiniCPM-o 4.5 serving with the optimized deploy config:

```bash
export HF_HUB_OFFLINE=1
export OMNI_TALKER_S0SPEC_K=14
export OMNI_TALKER_SCHED_K=12
export OMNI_TALKER_LOCAL_STEPS=12
export OMNI_TJS_STOP=1

vllm serve /path/to/MiniCPM-o-4_5 \
  --omni \
  --port 8091 \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml
```

For benchmark work, record the exact model path, commit, deploy config,
hardware, concurrency, prompt set, warmup policy, and quality gates. The
numbers above come from preserved branch evidence and commit messages; rerun
the workload on the target Ascend server before reporting fresh performance.

## Evidence Boundaries

- Final-stack RTF and first-packet numbers are cumulative. They combine graph,
  scheduling, cache, prewarm, and device-side changes.
- H1, K14, K12, T44, and TJS1 have separate recorded A/B evidence. Their
  latencies should not be added together as one total improvement.
- `enable_static_kernel` enables backend capability. It is not a user-authored
  Ascend C device kernel.
- Tagged raw-bytes handoff reduces Python object materialization. It is not
  shared-memory zero-copy.
- TJS1 changes CFM solver-step semantics, so it must be discussed with WER/SIM
  quality gates.

## Upstream

This repository is a fork and optimization branch of
[vLLM-Omni](https://github.com/vllm-project/vllm-omni), which extends vLLM for
omni-modality model inference and serving across text, image, audio, video, and
diffusion workloads.

Refer to upstream resources for general usage:

- [Documentation](https://vllm-omni.readthedocs.io/en/latest/)
- [Supported models](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)
- [vLLM project](https://github.com/vllm-project/vllm)

## Citation

If you use vLLM-Omni for research, cite the upstream paper:

```bibtex
@article{yin2026vllmomni,
  title={vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models},
  author={Peiqi Yin, Jiangyun Zhu, Han Gao, Chenguang Zheng, Yongxiang Huang, Taichang Zhou, Ruirui Yang, Weizhi Liu, Weiqing Chen, Canlin Guo, Didan Deng, Zifeng Mo, Cong Wang, James Cheng, Roger Wang, Hongsheng Liu},
  journal={arXiv preprint arXiv:2602.02204},
  year={2026}
}
```

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
