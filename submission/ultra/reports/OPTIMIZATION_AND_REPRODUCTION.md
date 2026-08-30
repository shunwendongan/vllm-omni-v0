# Ultra optimization and reproduction report

## Current six-level status

**Level 3 — local validation.** Ultra has team-reported performance on one A2,
but the official target is one A3/910C and the Ultra-specific full quality and
Demo gates are incomplete. It is not marked “complete accuracy” or “submittable.”

## Frozen contract

- Model: `OpenBMB/MiniCPM-o-4_5`.
- Candidate code base: `687af3ad5c66425ab77072cd4964164192897579`.
- Official evaluator: `minicpm-challenge@ecd9d99da0c124331861890e0371e66a01cddaa5`.
- Official image/hardware: `quay.io/ascend/vllm-omni:v0.25.0-a3`, one A3/910C.
- API: OpenAI-compatible chat completions with text and optional 24 kHz mono speech.
- Official configuration: the frozen files in `configs/`; candidate YAML is not
  substituted during evaluation.
- Ranking workload: Seed-TTS Chinese, c=1, 32 prompts, two warmups.
- Quality gates: Daily 0.775, Video-MME 0.670, WER 0.0156, SIM 0.689.

## Implementation areas present in the Ultra source

The repository's `docs/ultra/` design notes cover these implementation and
measurement areas:

- tensor handoff and FreeToken transfer;
- Talker output overlap and first/steady audio chunks;
- Code2Wav batch-one low-copy, early setup, and NPU Graph paths;
- Stage-0 TTFT work;
- numerical validation matrices and timeline measurement correctness;
- the MiniCPM-o 4.5 A3 contract.

These files describe hypotheses, code paths, safety conditions, and validation
methods. This submission report does **not** assign per-optimization speedups
without raw matched A/B evidence. Existing baseline capabilities such as the
three-stage pipeline, async chunks, and configured Stage-0/1 graphs are not
relabelled as Ultra improvements.

## Evidence separation

- Measured fact: the team supplied one A2 best-run row (RTF 0.56, TTFP 607.11
  ms, TTFT 376.17 ms, E2EL 2325.52 ms).
- Trace-supported inference: none is promoted in this package before trace files
  are imported.
- Untested on target: A3/910C latency, throughput, memory, full accuracy, Demo,
  stability, and zero-manual official-image reproduction.

## Installation and source identity

```bash
cd /workspace/vllm-competition/src/ultra
bash submission/ultra/scripts/install_candidate.sh
```

The installer refuses a dirty tree by default, verifies the expected code-base
commit is an ancestor, runs `pip install -e . --no-build-isolation --no-deps`,
and proves that `vllm_omni.__file__` resolves inside the candidate source.

## Full reproduction

```bash
bash submission/ultra/scripts/run_accuracy.sh
ACCURACY_SUMMARY_PATH=/path/to/summary.json \
  bash submission/ultra/scripts/run_performance.sh
bash submission/ultra/scripts/run_demo.sh
```

After Demo finalization, use `collect_remote_evidence.sh` and
`build_submission_tar.sh`. Each experiment records command lines, environment,
candidate and official commits, configuration SHA256, raw JSON, and logs in a
new run directory.

## Fallback and exclusions

The packaging PR adds evidence infrastructure only; it does not change inference
runtime behavior. Runtime fallbacks remain those of the candidate commit. The
archive excludes model weights, datasets, caches, virtual environments, generated
accuracy WAV corpora, and any material copied from another team's template.

## Remaining work

1. repair or replace the local Daily-Omni conversion so it contains the complete
   official row set;
2. run all four quality gates on Ultra and record the accepted WavLM/SV identity;
3. import all three c1 rounds and both concurrency guardrails, plus matched Slow;
4. complete and record the standard Gradio acceptance path;
5. reproduce in the official A3 image and update status monotonically.
