# Ultra performance report

## Outcome

The team has reported one best A2 run, but the raw files and all three formal
rounds are not yet present in this repository. Status is therefore
`EVIDENCE_PENDING_UPLOAD`, not submission-ready.

## Measured A2 fact currently available

| Hardware | Workload | RTF | TTFP | TTFT | E2EL |
|---|---|---:|---:|---:|---:|
| one Atlas A2 / 910B3 | Seed-TTS zh, 32 prompts, c=1, 2 warmups, best of a reported 3 runs | 0.56 | 607.11 ms | 376.17 ms | 2325.52 ms |

Reported remote source:
`/workspace/vllm-competition/results/ultra/performance/seedtts-c1-32x3/20260830-123645/summary.json`.

This row is retained as a measured team-supplied fact. It is not used to infer
the distribution, stability, matched speedup, or A3 result until the raw JSON is
collected.

## Official A3 reference — cross-hardware only

| Hardware | c / prompts | RTF | TTFP | TTFT | E2EL |
|---|---|---:|---:|---:|---:|
| one Atlas A3 / 910C | 1 / 32 | 0.4423 | 986.4666 ms | 333.2633 ms | 1857.2154 ms |

The A2 and A3 rows are not a valid source/candidate A/B. No result is divided by
two, multiplied by two, or converted between these hardware generations.

## Required repeated-run matrix

The checked-in runner performs:

- ranking unit: c=1, 32 prompts, 2 warmups, 3 formal repetitions;
- non-ranking guardrail: c=4, 64 prompts, 2 warmups, 1 repetition;
- non-ranking guardrail: c=8, 128 prompts, 2 warmups, 1 repetition.

For the ranking unit, the report must show every round plus mean, median, minimum,
maximum, and range for RTF, TTFP, TTFT, and E2EL. It also records successful and
failed requests, audio-metric availability, request throughput, service health,
and sampled resource output. A profiling run is never substituted for a formal
timing run.

## Matched Slow versus Ultra A/B

Not yet verified. To claim a source-level performance improvement, Slow and
Ultra must run on the same A2, model, official YAML, dataset order, output
contract, warmup, repetitions, and measurement scope. Different old runs are
diagnostic only.

## Reproduction

```bash
ACCURACY_SUMMARY_PATH=/path/to/ultra/accuracy/summary.json \
CANDIDATE_LABEL=ultra \
CANDIDATE_SRC=/workspace/vllm-competition/src/ultra \
  bash submission/ultra/scripts/run_performance.sh

# Repeat with CANDIDATE_LABEL=slow and the Slow source only after confirming the
# identical official contract. Keep its output in a separate run directory.
```

`BYPASS_ACCURACY_GATE=1` exists only for diagnostics. The resulting summary is
explicitly marked `NOT_RANKING_ELIGIBLE`.

## Resource and exception status

Peak NPU memory and resource distribution are pending the collected monitor log.
No failure rate or audio decode rate is claimed until the raw files are imported.
