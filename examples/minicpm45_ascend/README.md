# MiniCPM-o 4.5 — Ascend A3 competition dev/test framework

Versioned launch + benchmark scripts for the MiniCPM-o 4.5 competition submission
on the Ascend A3 server. Everything here is meant to be **committed and reused**
as the reproducible evidence trail for the race: how the server is started
(official spec), and how the three benchmarks (Seed-TTS, Daily-Omni, Video-MME)
are run and scored.

## Layout

| File | Purpose |
|---|---|
| `serve.sh` | Official-spec serve launcher (3-stage, FP16 + 4-step Code2Wav, Stage-0 32 tokens, port 8091). |
| `bench_seedtts.sh` | Seed-TTS performance + WER benchmark (32 prompts, concurrency 1, Whisper scoring). |
| `bench_dailyomni.sh` | Daily-Omni accuracy benchmark (1197 prompts, concurrency 10). |
| `bench_videomme.sh` | Video-MME accuracy benchmark (2700 prompts, concurrency 4). |

Each script is parameterized through environment variables with sane defaults and
never modifies vLLM-Omni core code.

## Environment requirements

- Ascend A3 NPU server (`openlibing-DevEnv-678907`), CANN + vLLM-Ascend + vLLM-Omni installed.
- vLLM-Omni repo checkout at `/vllm-workspace/vllm-omni` on branch `optimization-exploration`.
- `vllm` CLI on `PATH` (from vLLM-Omni, so `bench serve --omni` registers the
  `seed-tts` / `daily-omni` / `videomme` datasets).
- MiniCPM-o 4.5 checkpoint locally at `/root/models/MiniCPM-o-4_5`.
- Seed-TTS eval extras for WER scoring: `pip install 'vllm-omni[dev]'`
  (provides Whisper / jiwer / librosa).
- **No internet to HuggingFace Hub** — every script exports `HF_HUB_OFFLINE=1`;
  all model weights and datasets are local.

## Data paths

| Benchmark | Data | Location |
|---|---|---|
| Seed-TTS | Seed-TTS eval set (en split) | `/root/seed-tts-eval/seedtts_testset` |
| Daily-Omni | QA JSON + Videos | `/workspace/vllm-omni-data/daily-omni/{qa.json,Videos}` |
| Video-MME | parquet + video frames | `/workspace/vllm-omni-data/videomme/videomme/test-00000-of-00001.parquet`, `/workspace/vllm-omni-data/videomme/video` |

Override any of these with `DATASET_PATH` / `QA_JSON` / `VIDEO_DIR` /
`VIDEOMME_PARQUET` / `VIDEOMME_VIDEO_DIR`.

## Usage

### 1. Start the server (official spec)

```bash
cd /vllm-workspace/vllm-omni/examples/minicpm45_ascend
./serve.sh
```

- Runs `vllm serve /root/models/MiniCPM-o-4_5 --omni --deploy-config
  vllm_omni/deploy/minicpmo_4_5.yaml --port 8091 ...`.
- The deploy config pins the competition optimizations:
  `token2wav_float16=true` (FP16 Code2Wav), `token2wav_n_timesteps=4`
  (4-step CFM), and Stage-0 `max_tokens=32`.
- `serve.sh` **never** kills an existing serve: if port 8091 is already
  listening it exits with status 0 and leaves the running server untouched.

> The ongoing Daily-Omni baseline on port 8091 is one such live server; do not
> restart or kill it. If you need a second instance for experiments, run
> `PORT=8092 ./serve.sh` instead.

### 2. Run the benchmarks

Run each against the live server (defaults target port 8091):

```bash
# Seed-TTS: perf + WER (RTF / ttft / tpot / audio_rtf ... + Whisper WER)
./bench_seedtts.sh

# Daily-Omni: MCQ accuracy
./bench_dailyomni.sh

# Video-MME: MCQ accuracy
./bench_videomme.sh
```

Each writes its results JSON to the vLLM bench `--result-dir`/`--result-filename`
default (or wherever the `bench serve` saving flags point), e.g.
`Daily-Omni` accuracy under `daily_omni_accuracy`, `Video-MME` under
`videomme_accuracy`, and Seed-TTS WER under the `seed_tts_wer_*` keys.

### Parameter overrides

Every script reads env vars with defaults, so you can tune without editing:

| Variable | Default | Applies to |
|---|---|---|
| `PORT` | `8091` | all |
| `MODEL` | `openbmb/MiniCPM-o-4_5` (serve.sh: `/root/models/MiniCPM-o-4_5`) | all |
| `NUM_PROMPTS` | 32 / 1197 / 2700 | seed-tts / daily-omni / videomme |
| `MAX_CONCURRENCY` | 1 / 10 / 4 | seed-tts / daily-omni / videomme |
| `NUM_WARMUPS` | 3 / 1 / 1 | seed-tts / daily-omni / videomme |
| `DATASET_PATH` | `/root/seed-tts-eval/seedtts_testset` | bench_seedtts |
| `QA_JSON`, `VIDEO_DIR` | daily-omni paths under `/workspace/vllm-omni-data` | bench_dailyomni |
| `VIDEOMME_PARQUET`, `VIDEOMME_VIDEO_DIR` | videomme paths under `/workspace/vllm-omni-data` | bench_videomme |
| `PERCENTILE_METRICS` | seed-tts: `ttft,tpot,itl,e2el,audio_ttfp,audio_rtf`; others `ttft,tpot,itl,e2el` | all |

Example:

```bash
PORT=8092 NUM_PROMPTS=64 ./bench_seedtts.sh
```

Extra flags can be appended after `--`:

```bash
./bench_videomme.sh -- --result-dir /workspace/bench-results --result-filename videomme.json
```

## Notes

- The three benchmark commands mirror the validated MiniCPM-o 4.5 recipes:
  Daily-Omni `minicpm-interleave` + `--daily-omni-input-mode all`, Video-MME
  `minicpm-frames` (max 96 frames), Seed-TTS `use_tts_template` with
  text+audio modalities.
- All scripts `set -euo pipefail`; a failed bench aborts loudly rather than
  producing partial results.
- See `docs/ascend_performance_report.md` for the results template + scoring
  baseline and reproduction notes.
