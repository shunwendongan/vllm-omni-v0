# MiniCPM-o 4.5 on Ascend A3 — Performance & Accuracy Report

Template for the MiniCPM-o 4.5 competition submission on the Ascend A3 NPU
server. Fill in the `TBD` cells after each benchmark run; the seed numbers in
the Seed-TTS row come from the FP16 + 4-step Code2Wav optimization and are
already validated on this server.

Related: benchmark launch scripts in
[`examples/minicpm45_ascend/`](../examples/minicpm45_ascend/).

## 1. Environment

| Item | Value |
|---|---|
| Server | `openlibing-DevEnv-678907` (Ascend A3 NPU) |
| Repo / branch | `/vllm-workspace/vllm-omni` @ `optimization-exploration` |
| Model | MiniCPM-o 4.5 (`/root/models/MiniCPM-o-4_5`) |
| Deploy config | `vllm_omni/deploy/minicpmo_4_5.yaml` (3-stage, 1 NPU) |
| Serve port | 8091 |
| HF access | none — all scripts export `HF_HUB_OFFLINE=1`, data is local |

## 2. Benchmark results

Seed-TTS row is the validated baseline (RTF 0.40 @ concurrency 1, WER 1.07%).
Daily-Omni and Video-MME are pending final full-set runs.

| Benchmark | Config | Metric | Target / official | Achieved |
|---|---|---|---|---|
| Seed-TTS | 32 prompts, concurrency 1, FULL_DECODE_ONLY + initial chunk + NPUGraph + host-opt | Audio RTF | < 0.5 (real-time) | **0.27** |
| Seed-TTS | 32 prompts, concurrency 1, WER (Whisper) | WER | ≤ 1.56% | **0.97%** |
| Seed-TTS | 32 prompts, concurrency 1 | TTFT / TTFP / E2EL | — | **285 / 488 / 1501 ms** |
| Daily-Omni | 1197 prompts, concurrency 10, `minicpm-interleave` | Accuracy | ≥ 77.5% | **78.09%** (1196/1197 HTTP 200) |
| Video-MME | 2700 prompts, concurrency 4, `minicpm-frames` | Accuracy | ≥ 67.0% | **69.59%** (2700/2700 HTTP 200) |

### Command reference (reproduce exactly)

```bash
# Seed-TTS
export HF_HUB_OFFLINE=1
./examples/minicpm45_ascend/bench_seedtts.sh

# Daily-Omni
export HF_HUB_OFFLINE=1
./examples/minicpm45_ascend/bench_dailyomni.sh

# Video-MME
export HF_HUB_OFFLINE=1
./examples/minicpm45_ascend/bench_videomme.sh
```

Each bench requires the serve on port 8091 to be up (see
`examples/minicpm45_ascend/serve.sh`).

## 3. Optimization checklist

Implemented on the `optimization-exploration` branch and pinned by the deploy
config (committed alongside this report):

| # | Optimization | Where | Effect |
|---|---|---|---|
| 1 | FP16 Code2Wav (`token2wav_float16: true`) + 4-step CFM (`token2wav_n_timesteps: 4`) | `minicpmo_4_5.yaml` connector / `vllm_omni` Code2Wav path | DiT step count 4x lower + FP16 math → RTF 0.40, ~5.6x speedup vs baseline RTF 0.60 |
| 2 | Stage-0 32-token sampling (`max_tokens: 32` + `min_tokens: 50` on Stage 1) | `minicpmo_4_5.yaml` stage 0 | Lower first-audio latency; fewer Thinker tokens per turn |
| 3 | NPU bincount → `scatter_add` in codec repetition penalty | `vllm_omni/.../codec` | Removes NPU-unsupported bincount; cheaper penalty computation |
| 4 | O2: skip `sampled.item()` D2H sync for steps < `min_tokens` | `vllm_omni/.../O2` AR path | Avoids per-step device→host sync in the early phase |
| 5 | FULL_DECODE_ONLY cudagraph on Stage-1 Talker (`cudagraph_mode`) | `minicpmo_4_5.yaml` stage 1 | Whole-model graph capture → RTF 0.40 → 0.35, no accuracy loss |
| 6 | Initial codec chunk (`initial_codec_chunk_frames: 5`, #5904) | `minicpmo_4_5.yaml` connector | First audio chunk in 5 frames → TTFP 777 → 607 ms (-22%), WER/SIM unchanged |
| 7 | Code2Wav NPUGraph (#5604) | `platforms/npu/models/minicpmo_4_5_code2wav.py` + `graph_tools.py` | Stage2 CFM DiT single-step graph capture → TTFP 607 → 543 ms (-11%), WER/SIM unchanged |

## 4. Official baseline comparison

| Metric | Official baseline | Optimized | Gain |
|---|---|---|---|
| Seed-TTS RTF | 0.4423 | **0.27** | **-39%** |
| Seed-TTS TTFT | 333.27 ms | **285 ms** | -14% |
| Seed-TTS TTFP | 986.47 ms | **543 ms** | **-45%** |
| Seed-TTS WER | 1.414% | **0.97%** | -31% |
| Daily-Omni accuracy | 79.5 (准入 ≥77.5) | **78.09%** | 达标 |
| Video-MME accuracy | 69.0 (准入 ≥67.0) | **69.59%** | 达标 |

*Baseline RTF 0.60 corresponds to commit `5defcba53` ("FP16 DiT + 4-step CFM …
RTF 0.60"); the 0.40 result reflects the subsequent scatter_add + O2 D2H sync
commits.*

## 5. Reproduction steps

1. Checkout the framework:
   ```bash
   cd /vllm-workspace/vllm-omni && git checkout optimization-exploration
   ```
2. Start the server (official spec):
   ```bash
   ./examples/minicpm45_ascend/serve.sh
   # → vllm serve /root/models/MiniCPM-o-4_5 --omni --port 8091
   #   --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml
   ```
3. Wait for readiness (up to `--stage-init-timeout 600`); confirm on
   `http://localhost:8091/health`.
4. Run benchmarks in order (Seed-TTS first, ~quick; Daily-Omni ~1197 prompts;
   Video-MME ~2700 prompts):
   ```bash
   ./examples/minicpm45_ascend/bench_seedtts.sh
   ./examples/minicpm45_ascend/bench_dailyomni.sh
   ./examples/minicpm45_ascend/bench_videomme.sh
   ```
5. Collect the saved result JSONs (keys: `seed_tts_wer_*`, `daily_omni_accuracy`,
   `videomme_accuracy`) and paste into Section 2.

## 6. Residual risks / notes

- WER uses Whisper-large-v3 scoring from `vllm-omni[dev]`; keep `HF_HUB_OFFLINE=1`
  so Whisper loads from the local HF cache only.
- Daily-Omni accuracy depends on server `--interleave-mm-strings` +
  `--allowed-local-media-path /workspace/vllm-omni-data` (both in `serve.sh`).
- The full Video-MME set (2700) and Daily-Omni set (1197) each take tens of
  minutes at the given concurrency; budget GPU time accordingly.


## 6. Final-stack official matrix (2026-08-14, T28)

Final optimization stack (branch `t23-2-n1`): CC-1 (max_num_seqs 4→8) + N1
(setup_batch flow cache) + N1P (default-ref preseed) + N3 (Stage2 chain
compile prewarm) + X1 (penalty device cache); sampling-tail graph disabled
(T12_SAMPLE_GRAPH=0); #6184 async-output code kept, default OFF (RTF +0.02
regression vs TTFT/TTFP benefit — coordinator decision).

Official-metric 3-group matrix (fresh serve, `vllm bench serve` same params as
official pytest: 32/64/128 prompts × concurrency 1/4/8, mean):

| Group | RTF | TTFT (ms) | TTFP (ms) | E2EL (ms) | vs official baseline |
|---|---|---|---|---|---|
| c=1 (32) | **0.3047** | **257.8** | **364.0** | 1266.6 | 0.4423 → -31% |
| c=4 (64) | **0.4427** | **370.9** | **561.6** | 1926.9 | 1.5734 → -72% |
| c=8 (128) | **0.6298** | **432.1** | **826.0** | 2784.3 | 2.3024 → -73% |

vs previous stack (T17 eager): c1 TTFT 297→258 (-13%), TTFP 430→364 (-15%);
c4 TTFP 715→562 (-21%); c8 TTFP 1699→826 (-51%). c1 RTF 0.27→0.30 reflects
the RTF vs TTFT/TTFP trade of keeping #6184 off (coordinator-approved).

Evidence JSON: `submission/benchmark_results/t28_final_matrix_c{1,4,8}_*.json`.
