# Ultra accuracy report

## Outcome

**NOT_VERIFIED.** No Slow result or proxy-only score is used as Ultra evidence.
The committed JSON is a status record, not a claimed accuracy result.

## Official gates and frozen workload

| Suite | Parameters | Complete-coverage gate | Quality gate |
|---|---|---|---|
| Daily-Omni | `temperature=0`, `output_len=512`, input `all`, pack `minicpm-interleave` | expected official/local row count resolved; zero HTTP failures or drops | accuracy ≥ 0.775 |
| Video-MME | no subtitles, `temperature=0`, `output_len=128`, `max_frames=96`, `minicpm-frames`, duration `all` | 2700/2700; zero HTTP failures | accuracy ≥ 0.670 |
| Seed-TTS | Chinese split, deterministic TTS, WER evaluation and speaker similarity | 2020/2020 WER and 2020/2020 SIM; zero request/PCM/ASR/embedding failures | WER ≤ 0.0156 and SIM ≥ 0.689 |

The local Daily-Omni conversion previously produced 1196 QA rows while the
expected official set is 1197. `run_accuracy.sh` stops before inference when
that mismatch remains; it is not legal to score 1196 rows and call the result
complete.

## Speaker-similarity protocol

At official commit `ecd9d99d...`, the in-tree evaluator describes the default
`microsoft/wavlm-base-plus` mean-pooled embedding cosine as a speaker-similarity
**proxy**, and distinguishes it from Seed-TTS's fine-tuned UniSpeech/WavLM-SV
checkpoint. Every accepted result must therefore record:

- the exact `SEED_TTS_WAVLM_MODEL` value or path;
- its revision or file SHA256 when local;
- 2020 evaluated reference/synthesis pairs and zero embedding failures;
- whether the result is the official challenge evaluator's accepted contract.

An unnamed WavLM score cannot be promoted to an official-equivalent ASV claim.

## Reproduction

```bash
cd /workspace/vllm-competition/src/ultra

COMP_ROOT=/workspace/vllm-competition \
MODEL_PATH=/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5 \
DATA_ROOT=/workspace/vllm-competition/data \
OFFICIAL_SRC=/workspace/vllm-competition/src/official-minicpm-challenge \
CANDIDATE_SRC=/workspace/vllm-competition/src/ultra \
SEED_TTS_WAVLM_MODEL=/path/to/the/accepted/wavlm-checkpoint \
  bash submission/ultra/scripts/run_accuracy.sh
```

The script starts a clean Ultra service using the frozen official deploy YAML,
runs each suite independently, saves per-item records, checks complete coverage,
and writes one aggregate `summary.json`. Generated 2020 WAVs remain remote and
are excluded from the final tar.

## Evidence to import

- all three raw benchmark JSON files;
- `summary.json`, service log, commands, manifest, and environment versions;
- dataset counts and dataset hashes/revisions;
- the exact WavLM/SV checkpoint identity and failures count.
