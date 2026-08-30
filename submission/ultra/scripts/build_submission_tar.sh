#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
init_ultra_submission_env

ALLOW_NOT_READY=${ALLOW_NOT_READY:-0}
RUN_INSTALL_CHECK=${RUN_INSTALL_CHECK:-0}
OUTPUT_DIR=${OUTPUT_DIR:-$COMP_ROOT/submission-artifacts}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-}

SUBMISSION_ROOT="$CANDIDATE_SRC/submission/ultra"
require_dir "$CANDIDATE_SRC/.git"
require_dir "$SUBMISSION_ROOT"
require_executable "$TEST_PY"
require_file "$FROZEN_DEPLOY_CONFIG"
require_file "$SUBMISSION_ROOT/configs/official-performance-config.json"

git -C "$CANDIDATE_SRC" merge-base --is-ancestor \
  "$EXPECTED_CODE_BASE_COMMIT" HEAD || \
  die "candidate HEAD does not derive from $EXPECTED_CODE_BASE_COMMIT"
if test -n "$(git -C "$CANDIDATE_SRC" status --porcelain)"; then
  git -C "$CANDIDATE_SRC" status --short >&2
  die "candidate worktree must be clean before packaging"
fi

test "$(sha256_file "$FROZEN_DEPLOY_CONFIG")" = \
  "b3c35aad87ddeba64781b3833b64c4a7132eaf88fc29002bb018e6b84fdb76ed" || \
  die "frozen deploy config hash mismatch"
test "$(sha256_file "$SUBMISSION_ROOT/configs/official-performance-config.json")" = \
  "01ace1ad6e06823be75b97d85cede0a5542c097c7493a9a06a77537128aff1d2" || \
  die "frozen performance config hash mismatch"

if test -n "$EVIDENCE_ROOT"; then
  require_dir "$EVIDENCE_ROOT"
  ACCURACY_SUMMARY_PATH=${ACCURACY_SUMMARY_PATH:-$EVIDENCE_ROOT/accuracy/summary.json}
  PERFORMANCE_SUMMARY_PATH=${PERFORMANCE_SUMMARY_PATH:-$EVIDENCE_ROOT/performance/summary.json}
  DEMO_SUMMARY_PATH=${DEMO_SUMMARY_PATH:-$EVIDENCE_ROOT/demo/demo-summary.json}
else
  ACCURACY_SUMMARY_PATH=${ACCURACY_SUMMARY_PATH:-$SUBMISSION_ROOT/results/accuracy-summary.json}
  PERFORMANCE_SUMMARY_PATH=${PERFORMANCE_SUMMARY_PATH:-$SUBMISSION_ROOT/results/performance-summary.json}
  DEMO_SUMMARY_PATH=${DEMO_SUMMARY_PATH:-$SUBMISSION_ROOT/results/demo-summary.json}
fi
require_file "$ACCURACY_SUMMARY_PATH"
require_file "$PERFORMANCE_SUMMARY_PATH"
require_file "$DEMO_SUMMARY_PATH"

set +e
READINESS=$(
  "$TEST_PY" - "$ACCURACY_SUMMARY_PATH" "$PERFORMANCE_SUMMARY_PATH" "$DEMO_SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

accuracy = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
performance = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
demo = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
reasons = []

if accuracy.get("status") != "PASS" or accuracy.get("coverage_ok") is not True:
    reasons.append("accuracy status/coverage")
if accuracy.get("daily_omni", {}).get("gate_pass") is not True:
    reasons.append("Daily-Omni gate")
if accuracy.get("videomme", {}).get("gate_pass") is not True:
    reasons.append("Video-MME gate")
if accuracy.get("seed_tts", {}).get("wer_gate_pass") is not True:
    reasons.append("Seed-TTS WER gate")
if accuracy.get("seed_tts", {}).get("asv_sim_gate_pass") is not True:
    reasons.append("Seed-TTS ASV/SIM gate")

ranking = performance.get("ranking_unit", {}).get("formal_runs", [])
guardrails = performance.get("guardrails_non_ranking", {})
if performance.get("status") != "PASS":
    reasons.append("performance status")
if len(ranking) != 3 or not all(row.get("valid") is True for row in ranking):
    reasons.append("three valid c1 formal runs")
if not all(guardrails.get(key) and guardrails[key][0].get("valid") is True for key in ("c4-p64", "c8-p128")):
    reasons.append("c4/c8 guardrails")

required_modalities = {"text", "image", "audio", "video"}
if demo.get("status") != "PASS":
    reasons.append("Demo status")
if int(demo.get("interaction_count", 0) or 0) < 10:
    reasons.append("ten Demo interactions")
if not required_modalities.issubset(set(demo.get("modalities_seen", []))):
    reasons.append("Demo modality coverage")
if demo.get("service_stable") is not True:
    reasons.append("Demo stability")

if reasons:
    print("NOT_READY:" + "; ".join(reasons))
    raise SystemExit(2)
print("READY")
PY
)
readiness_rc=$?
set -e

if test "$readiness_rc" -eq 0; then
  PACKAGE_STATUS=READY
  STATUS_SUFFIX=
else
  PACKAGE_STATUS=NOT_READY
  STATUS_SUFFIX=-NOT_READY
  test "$ALLOW_NOT_READY" = "1" || \
    die "$READINESS; set ALLOW_NOT_READY=1 only to build a diagnostic archive"
fi

mkdir -p "$OUTPUT_DIR"
SOURCE_HEAD=$(git -C "$CANDIDATE_SRC" rev-parse HEAD)
SOURCE_SHORT=$(git -C "$CANDIDATE_SRC" rev-parse --short=12 HEAD)
PACKAGE_NAME="shunwendongan-vllm-omni-ultra-${SOURCE_SHORT}${STATUS_SUFFIX}"
ARCHIVE_PATH="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
test ! -e "$ARCHIVE_PATH" || die "refusing to overwrite archive: $ARCHIVE_PATH"

WORK_DIR=$(mktemp -d "$OUTPUT_DIR/.ultra-package.XXXXXX")
VERIFY_DIR=$(mktemp -d "$OUTPUT_DIR/.ultra-verify.XXXXXX")
cleanup() {
  rm -rf "$WORK_DIR" "$VERIFY_DIR"
}
trap cleanup EXIT INT TERM
PACKAGE_ROOT="$WORK_DIR/$PACKAGE_NAME"

mkdir -p \
  "$PACKAGE_ROOT/01_code/vllm-omni" \
  "$PACKAGE_ROOT/01_code/configs" \
  "$PACKAGE_ROOT/02_benchmark_results/daily_omni" \
  "$PACKAGE_ROOT/02_benchmark_results/seed_tts_accuracy" \
  "$PACKAGE_ROOT/02_benchmark_results/video_mme" \
  "$PACKAGE_ROOT/03_performance_report/raw" \
  "$PACKAGE_ROOT/04_demo" \
  "$PACKAGE_ROOT/05_optimization_report" \
  "$PACKAGE_ROOT/docs"

copy_filtered_tree() {
  local source_dir=$1
  local target_dir=$2
  mkdir -p "$target_dir"
  tar -C "$source_dir" \
    --exclude='./.venv' \
    --exclude='./venv' \
    --exclude='./env' \
    --exclude='./results' \
    --exclude='./data' \
    --exclude='./datasets' \
    --exclude='./cache' \
    --exclude='./.cache' \
    --exclude='./work/perf-evidence' \
    --exclude='./ultra-timeline' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.wav' \
    -cf - . | tar -C "$target_dir" -xf -
}

copy_filtered_tree "$CANDIDATE_SRC" "$PACKAGE_ROOT/01_code/vllm-omni"

cp "$SUBMISSION_ROOT/README.md" "$PACKAGE_ROOT/README.md"
cp "$SUBMISSION_ROOT/README.md" "$PACKAGE_ROOT/01_code/README.md"
cp "$SUBMISSION_ROOT/reports/ACCURACY_REPORT.md" "$PACKAGE_ROOT/02_benchmark_results/README.md"
cp "$SUBMISSION_ROOT/reports/PERFORMANCE_REPORT.md" "$PACKAGE_ROOT/03_performance_report/README.md"
cp "$SUBMISSION_ROOT/reports/DEMO_VALIDATION.md" "$PACKAGE_ROOT/04_demo/README.md"
cp "$SUBMISSION_ROOT/reports/OPTIMIZATION_AND_REPRODUCTION.md" "$PACKAGE_ROOT/05_optimization_report/README.md"
cp "$SUBMISSION_ROOT/SUBMISSION_CHECKLIST.md" "$PACKAGE_ROOT/docs/COMPLIANCE_CHECK.md"
cp "$SUBMISSION_ROOT/reports/TEAM_INFO.md" "$PACKAGE_ROOT/docs/team_info.md"
cp "$SUBMISSION_ROOT/configs/"* "$PACKAGE_ROOT/01_code/configs/"

for script in \
  _common.sh \
  install_candidate.sh \
  serve.sh \
  bench_daily_omni.sh \
  bench_seedtts_zh.sh \
  bench_videomme.sh \
  bench_perf_c1.sh \
  run_accuracy.sh \
  run_performance.sh \
  run_demo.sh \
  collect_remote_evidence.sh \
  build_submission_tar.sh; do
  cp "$SUBMISSION_ROOT/scripts/$script" "$PACKAGE_ROOT/01_code/$script"
  chmod +x "$PACKAGE_ROOT/01_code/$script"
done

cp "$CANDIDATE_SRC/examples/online_serving/minicpmo/gradio_demo.py" "$PACKAGE_ROOT/04_demo/gradio_demo.py"
cp "$CANDIDATE_SRC/examples/online_serving/minicpmo/run_gradio_demo.sh" "$PACKAGE_ROOT/04_demo/run_gradio_demo.sh"
chmod +x "$PACKAGE_ROOT/04_demo/run_gradio_demo.sh"

cp "$ACCURACY_SUMMARY_PATH" "$PACKAGE_ROOT/02_benchmark_results/accuracy-summary.json"
cp "$PERFORMANCE_SUMMARY_PATH" "$PACKAGE_ROOT/03_performance_report/raw/performance-summary.json"
cp "$DEMO_SUMMARY_PATH" "$PACKAGE_ROOT/04_demo/demo-summary.json"

if test -n "$EVIDENCE_ROOT"; then
  for suite in daily_omni seed_tts videomme; do
    source_suite="$EVIDENCE_ROOT/accuracy/$suite"
    case "$suite" in
      daily_omni) target_suite="$PACKAGE_ROOT/02_benchmark_results/daily_omni" ;;
      seed_tts) target_suite="$PACKAGE_ROOT/02_benchmark_results/seed_tts_accuracy" ;;
      videomme) target_suite="$PACKAGE_ROOT/02_benchmark_results/video_mme" ;;
    esac
    if test -d "$source_suite"; then
      tar -C "$source_suite" --exclude='*.wav' --exclude='./audio' --exclude='./audio/*' \
        -cf - . | tar -C "$target_suite" -xf -
    fi
  done
  if test -d "$EVIDENCE_ROOT/performance"; then
    tar -C "$EVIDENCE_ROOT/performance" -cf - . | tar -C "$PACKAGE_ROOT/03_performance_report/raw" -xf -
  fi
  if test -d "$EVIDENCE_ROOT/demo"; then
    tar -C "$EVIDENCE_ROOT/demo" -cf - . | tar -C "$PACKAGE_ROOT/04_demo" -xf -
  fi
  cp "$EVIDENCE_ROOT/manifest.json" "$PACKAGE_ROOT/docs/evidence-environment-manifest.json"
  cp "$EVIDENCE_ROOT/evidence-index.json" "$PACKAGE_ROOT/docs/evidence-index.json"
fi

# Accuracy WAV corpora are forbidden; a few validated Demo outputs are allowed.
if find "$PACKAGE_ROOT/02_benchmark_results" -type f -name '*.wav' -print -quit | grep -q .; then
  die "accuracy WAV files leaked into the package"
fi
if find "$PACKAGE_ROOT/01_code/vllm-omni" -type f \
  \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) \
  -print -quit | grep -q .; then
  die "model/checkpoint files leaked into the source package"
fi
require_dir "$PACKAGE_ROOT/01_code/vllm-omni/.git"

PACKAGE_ROOT="$PACKAGE_ROOT" PACKAGE_STATUS="$PACKAGE_STATUS" \
SOURCE_HEAD="$SOURCE_HEAD" READINESS="$READINESS" "$TEST_PY" - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["PACKAGE_ROOT"])
manifest = root / "MANIFEST.sha256"

with (root / "README.md").open("a", encoding="utf-8") as stream:
    stream.write("\n## Materialized archive status\n\n")
    stream.write(f"- Status: `{os.environ['PACKAGE_STATUS']}`\n")
    stream.write(f"- Source commit: `{os.environ['SOURCE_HEAD']}`\n")
    stream.write(f"- Gate detail: `{os.environ['READINESS']}`\n")

rows = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path == manifest:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append(f"{digest.hexdigest()}  {path.relative_to(root)}")
manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
PY

tar -czf "$ARCHIVE_PATH" -C "$WORK_DIR" "$PACKAGE_NAME"
tar -xzf "$ARCHIVE_PATH" -C "$VERIFY_DIR"
EXTRACTED_ROOT="$VERIFY_DIR/$PACKAGE_NAME"
for path in \
  README.md \
  01_code/vllm-omni/.git \
  02_benchmark_results \
  03_performance_report/raw \
  04_demo/gradio_demo.py \
  05_optimization_report/README.md \
  docs/COMPLIANCE_CHECK.md \
  MANIFEST.sha256; do
  test -e "$EXTRACTED_ROOT/$path" || die "archive verification missing: $path"
done

EXTRACTED_ROOT="$EXTRACTED_ROOT" "$TEST_PY" - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["EXTRACTED_ROOT"])
for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
    expected, relative = line.split("  ", 1)
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"checksum mismatch: {relative}")
print("MANIFEST verification: PASS")
PY
git -C "$EXTRACTED_ROOT/01_code/vllm-omni" fsck --full --no-dangling

if test "$PACKAGE_STATUS" = "READY" && test "$RUN_INSTALL_CHECK" != "1"; then
  die "a READY archive requires RUN_INSTALL_CHECK=1"
fi
if test "$RUN_INSTALL_CHECK" = "1"; then
  "$TEST_PY" -m venv --system-site-packages "$VERIFY_DIR/install-venv"
  "$VERIFY_DIR/install-venv/bin/python" -m pip install \
    --editable "$EXTRACTED_ROOT/01_code/vllm-omni" \
    --no-build-isolation \
    --no-deps \
    > "$VERIFY_DIR/install-check.log" 2>&1
  (
    cd "$EXTRACTED_ROOT/01_code/vllm-omni"
    "$VERIFY_DIR/install-venv/bin/python" -c \
      'from pathlib import Path; import vllm_omni; source=Path(vllm_omni.__file__).resolve(); expected=Path.cwd().resolve(); assert expected in source.parents; print(source)'
  ) >> "$VERIFY_DIR/install-check.log" 2>&1
fi

ARCHIVE_SHA=$(sha256_file "$ARCHIVE_PATH")
printf 'PACKAGE_STATUS=%s\n' "$PACKAGE_STATUS"
printf 'ARCHIVE_PATH=%s\n' "$ARCHIVE_PATH"
printf 'ARCHIVE_SHA256=%s\n' "$ARCHIVE_SHA"
printf 'SOURCE_COMMIT=%s\n' "$SOURCE_HEAD"
