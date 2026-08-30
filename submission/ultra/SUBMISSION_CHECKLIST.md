# Ultra submission checklist

Last reviewed: 2026-08-31.

| Gate | Requirement | Current status |
|---|---|---|
| Source identity | Candidate derives from `687af3ad...`, clean tree, editable install imports the candidate source | Pending remote replay |
| Official contract | Deploy YAML and performance JSON from `ecd9d99d...` | Frozen and checksummed |
| Daily-Omni | Full official set, no dropped/failed rows, accuracy at least 77.5% | **NOT_VERIFIED** |
| Daily row count | Resolve local 1196 versus expected official 1197 before scoring | **BLOCKING** |
| Video-MME | 2700 rows, no subtitle, 96 frames, accuracy at least 67.0% | **NOT_VERIFIED** |
| Seed-TTS WER | 2020 Chinese rows, zero failures, mean WER at most 1.56% | **NOT_VERIFIED** |
| Seed-TTS ASV/SIM | 2020 reference pairs, named WavLM/SV checkpoint, mean at least 0.689 | **NOT_VERIFIED** |
| Performance | c1/32 with two warmups and three formal runs; c4/64 and c8/128 guardrails | Evidence pending upload |
| Fair A/B | Slow and Ultra on the same A2, same official config and workload | **NOT_VERIFIED** |
| Demo | text/image/audio/video, streaming speech, 10 interactions, 24 kHz mono WAV, recording | **NOT_VERIFIED** |
| Stability | service survives all Demo interactions and guardrails | **NOT_VERIFIED** |
| Official target | single A3/910C in official image | Pending official reproduction |
| Archive | template-compatible layout, `.git`, SHA256, extraction test, `git fsck`, isolated editable install | Scripted; not run with final evidence |

## Promotion rule

The archive may omit the `NOT_READY` suffix and the PR may leave Draft only when:

1. all four accuracy gates pass on the same Ultra commit and official config;
2. every dataset has complete coverage with zero silent drops;
3. the performance report contains all three c1 rounds plus both guardrails and
   does not hide failures or only report the best round;
4. the standard Gradio Demo evidence passes all required modalities and ten
   consecutive interactions;
5. the final archive passes extraction, checksum, Git, and isolated install checks.

Slow-branch accuracy, a proxy score with an unidentified embedding model, or an
A2-to-A3 arithmetic conversion cannot satisfy these gates.
