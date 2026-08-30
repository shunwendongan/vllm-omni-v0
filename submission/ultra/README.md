# shunwendongan MiniCPM-o 4.5 Ultra submission

This directory is the reproducible evidence and packaging layer for the `Ultra`
candidate. It does not change the inference API, model protocol, official deploy
configuration, or official benchmark parameters.

Last reviewed: 2026-08-31.

## Current status

**NOT_READY — Draft PR only.** The source is an A2-tested submission candidate;
official A3/910C reproduction is pending. Ultra-specific full accuracy, the
required speaker-similarity evidence, the complete repeated-run performance
bundle, and the Demo recording have not yet been imported into this repository.

| Item | Frozen value or status |
|---|---|
| Team | `shunwendongan` |
| Candidate code base | `687af3ad5c66425ab77072cd4964164192897579` |
| Official test base | `minicpm-challenge@ecd9d99da0c124331861890e0371e66a01cddaa5` |
| Official image | `quay.io/ascend/vllm-omni:v0.25.0-a3` |
| Official hardware | one Atlas A3 / Ascend 910C NPU |
| Local hardware | one Atlas A2 / Ascend 910B3 NPU |
| Local performance evidence | upload pending; one reported best run is recorded in `results/performance-summary.json` |
| Accuracy | `NOT_VERIFIED` for Ultra |
| Demo | `NOT_VERIFIED` |

A2 results are never divided by two or converted into estimated A3 results.
The official A3 baseline is shown only as a cross-hardware reference.

## Repository contents

- `configs/` freezes the official commit, deploy YAML, performance matrix, and
  evidence schema.
- `scripts/` installs and verifies the candidate, runs full accuracy, runs the
  repeated performance matrix, starts/finalizes Demo validation, collects raw
  evidence, and builds the final archive.
- `results/` contains reviewed status summaries. Placeholder states are explicit;
  they are not benchmark results.
- `reports/` contains the accuracy, performance, Demo, and optimization reports.

## Reproduction order

All commands below run on the Ascend evaluation machine. Each script accepts
`COMP_ROOT`, `MODEL_PATH`, `DATA_ROOT`, `OFFICIAL_SRC`, `CANDIDATE_SRC`, and
`RUN_ID`. Every run creates a new directory and refuses to overwrite evidence.

```bash
cd /workspace/vllm-competition/src/ultra

bash submission/ultra/scripts/install_candidate.sh
bash submission/ultra/scripts/run_accuracy.sh

# Performance is blocked unless Ultra accuracy passed. A diagnostic-only bypass
# exists, but results produced with it are marked ineligible for ranking.
ACCURACY_SUMMARY_PATH=/path/to/ultra/accuracy/summary.json \
  bash submission/ultra/scripts/run_performance.sh

bash submission/ultra/scripts/run_demo.sh
```

After the manual Demo interactions and recording are complete, finalize the Demo
run and collect all evidence:

```bash
DEMO_ACTION=finalize \
DEMO_RUN_DIR=/path/to/demo/run \
DEMO_VIDEO_PATH=/path/to/demo-recording.mp4 \
  bash submission/ultra/scripts/run_demo.sh

ACCURACY_RESULT_DIR=/path/to/accuracy/run \
PERFORMANCE_RESULT_DIR=/path/to/performance/run \
DEMO_RESULT_DIR=/path/to/demo/run \
  bash submission/ultra/scripts/collect_remote_evidence.sh
```

## Final archive layout

`build_submission_tar.sh` materializes the directory organization of
`/Users/MacBook/Desktop/template_official` without copying that template's team
claims, source, benchmark JSON, or video:

```text
README.md
01_code/
02_benchmark_results/
03_performance_report/
04_demo/
05_optimization_report/
docs/
MANIFEST.sha256
```

The complete source repository under `01_code/vllm-omni/` includes `.git`.
Models, datasets, caches, virtual environments, and the 2020 generated accuracy
WAV files are excluded. Until every gate is passed, packaging requires
`ALLOW_NOT_READY=1` and emits a filename ending in `-NOT_READY.tar.gz`.

```bash
ALLOW_NOT_READY=1 \
EVIDENCE_ROOT=/path/to/collected/evidence \
  bash submission/ultra/scripts/build_submission_tar.sh
```

See `SUBMISSION_CHECKLIST.md` for the exact promotion gates.
