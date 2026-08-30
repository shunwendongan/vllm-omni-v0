# MiniCPM-o 4.5 Demo validation

## Outcome

**NOT_VERIFIED.** The standard Gradio path is scripted, but no Ultra Demo video,
interaction ledger, screenshots, or representative output WAVs have been
collected yet. Experimental Realtime Duplex is optional and cannot replace this
gate.

## Standard path

Backend:

```bash
vllm serve "$MODEL_PATH" \
  --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --deploy-config "$OFFICIAL_SRC/vllm_omni/deploy/minicpmo_4_5.yaml" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8099
```

Frontend:

```bash
python examples/online_serving/minicpmo/gradio_demo.py \
  --minicpmo45-api-base http://127.0.0.1:8099/v1 \
  --minicpmo45-model openbmb/MiniCPM-o-4_5 \
  --host 0.0.0.0 \
  --port 7862
```

`run_demo.sh` launches both from the Ultra source, freezes the service command,
checks health, and leaves the UI available until interrupted. On a Mac, open a
platform port mapping or an SSH tunnel:

```bash
ssh -N -L 7862:127.0.0.1:7862 -L 8099:127.0.0.1:8099 user@remote-host
```

Then browse to `http://127.0.0.1:7862` and use macOS screen recording.

## Required interaction ledger

Record at least ten consecutive interactions in `interactions.jsonl`. Every row
uses this contract:

```json
{"interaction_id":1,"input_modality":"text","output_text_nonempty":true,"streaming_audio":true,"output_audio_path":"/absolute/path/output.wav","service_healthy_after":true,"notes":""}
```

Across the ten rows, `input_modality` must cover `text`, `image`, `audio`, and
`video`. Validation requires non-empty text, streamed audio, a healthy service
after every interaction, and WAV outputs that decode as non-empty 24 kHz mono.

## Finalization

```bash
DEMO_ACTION=finalize \
DEMO_RUN_DIR=/path/printed/by/start \
DEMO_VIDEO_PATH=/path/to/mac-recording.mp4 \
DEMO_SCREENSHOT_DIR=/path/to/screenshots \
  bash submission/ultra/scripts/run_demo.sh
```

The finalizer verifies the ledger and WAV headers, copies a small representative
set of outputs, hashes the video/screenshots/outputs, and writes
`demo-summary.json`. It does not accept a video alone without the machine-readable
ledger.
