## Summary

Package the reproducible evidence, official-contract snapshots, remote runners,
Demo validation workflow, and official-template-compatible archive builder for
the `shunwendongan` MiniCPM-o 4.5 Ultra candidate.

This PR adds submission material only. It does not modify inference code, API
semantics, the candidate deploy YAML, model weights, or official test parameters.

## Frozen contract

- Candidate code base: `687af3ad5c66425ab77072cd4964164192897579`
- Official evaluator: `minicpm-challenge@ecd9d99da0c124331861890e0371e66a01cddaa5`
- Official target: one Atlas A3 / Ascend 910C in
  `quay.io/ascend/vllm-omni:v0.25.0-a3`
- Local evidence hardware: one Atlas A2 / Ascend 910B3

## Current evidence

- A2 team-reported best run, Seed-TTS zh c1/32, two warmups:
  RTF 0.56, TTFP 607.11 ms, TTFT 376.17 ms, E2EL 2325.52 ms.
- Raw repeated-run upload: pending.
- Ultra full accuracy: pending.
- Standard Gradio Demo recording and ten-interaction ledger: pending.

A2 metrics are not converted into estimated A3 values. Slow accuracy is not
reused as Ultra accuracy.

## Validation

- Shell syntax checks for every submitted script
- JSON parsing and official snapshot SHA256 checks
- Markdown path/link scan
- `git diff --check`
- diff-scoped code-quality sweep
- package extraction, SHA256, copied-source `git fsck`, and isolated editable
  install are implemented in the archive builder

## Review status

**Draft / NOT_READY.** Keep this PR in Draft until all four Ultra accuracy gates,
full coverage, all performance repetitions, stability guardrails, and the
standard Demo evidence are attached and reviewed. A3/910C official reproduction
is pending.
