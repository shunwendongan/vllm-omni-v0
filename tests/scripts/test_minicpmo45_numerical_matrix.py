# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from scripts.run_minicpmo45_numerical_matrix import (
    ARMS,
    PAIR_ORDER,
    _load_command_config,
    _render,
    build_schedule,
    create_evidence_root,
    summarize_json_results,
    validate_official_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_matrix_schedule_is_restart_isolated_bccbbc_for_each_candidate():
    schedule = build_schedule(["S8", "H10"])

    assert [spec.role for spec in schedule[:6]] == list(PAIR_ORDER)
    assert [spec.arm.name for spec in schedule[:6]] == ["B10", "S8", "S8", "B10", "B10", "S8"]
    assert [spec.arm.name for spec in schedule[6:]] == ["B10", "H10", "H10", "B10", "B10", "H10"]
    assert [spec.ordinal for spec in schedule] == list(range(12))


@pytest.mark.parametrize("candidates", [["B10"], ["unknown"], [], ["S8", "S8"]])
def test_matrix_schedule_rejects_invalid_candidates(candidates):
    with pytest.raises(ValueError):
        build_schedule(candidates)


def test_arm_grid_freezes_steps_and_precision():
    assert {name: (arm.steps, arm.flow_fp16) for name, arm in ARMS.items()} == {
        "B10": (10, False),
        "S8": (8, False),
        "S6": (6, False),
        "H10": (10, True),
        "H8": (8, True),
        "H6": (6, True),
    }


def test_evidence_root_refuses_overwrite(tmp_path):
    first = create_evidence_root(tmp_path, "fixed-run")
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        create_evidence_root(tmp_path, "fixed-run")


def test_official_contract_requires_chinese_fixed_order_and_1_4_8(tmp_path):
    config = [
        {
            "test_name": "test_minicpmo_4_5_challenge",
            "benchmark_params": [
                {
                    "dataset_name": "seed-tts",
                    "dataset_path": "seed/repo",
                    "seed_tts_locale": "zh",
                    "disable_shuffle": True,
                    "max_concurrency": [1, 4, 8],
                    "num_prompts": [32, 64, 128],
                }
            ],
        }
    ]
    path = tmp_path / "official.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    frozen = validate_official_config(path)

    assert frozen["locale"] == "zh"
    assert frozen["max_concurrency"] == [1, 4, 8]
    assert len(frozen["sha256"]) == 64

    config[0]["benchmark_params"][0]["seed_tts_locale"] = "en"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Chinese"):
        validate_official_config(path)


def test_command_template_is_argv_only_and_expands_evidence_values():
    command = _render(
        ["bench", "--arm", "{arm}", "--result-dir", "{benchmark_dir}", "--num-warmups", "{warmups}"],
        {"arm": "S8", "benchmark_dir": "/tmp/evidence/raw", "warmups": 2},
    )
    assert command == [
        "bench",
        "--arm",
        "S8",
        "--result-dir",
        "/tmp/evidence/raw",
        "--num-warmups",
        "2",
    ]


def test_command_config_accepts_official_runner_managed_server(tmp_path):
    path = tmp_path / "commands.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_command": [
                    "python",
                    "-m",
                    "pytest",
                    "tests/dfx/perf/scripts/run_benchmark.py",
                    "--test-config-file",
                    "{official_config}",
                ]
            }
        ),
        encoding="utf-8",
    )

    config = _load_command_config(path, dry_run=False)

    assert set(config) == {"benchmark_command"}


def test_command_config_rejects_half_explicit_server(tmp_path):
    path = tmp_path / "commands.json"
    path.write_text(
        json.dumps({"server_command": ["serve"], "benchmark_command": ["bench"]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provided together"):
        _load_command_config(path, dry_run=False)


def test_result_summary_preserves_all_repetitions(tmp_path):
    run_dir = tmp_path / "run"
    raw = run_dir / "benchmark" / "raw"
    raw.mkdir(parents=True)
    (raw / "one.json").write_text(
        json.dumps(
            {
                "dataset_name": "seed-tts",
                "max_concurrency": 1,
                "num_prompts": 32,
                "mean_audio_rtf": 0.4,
                "mean_audio_ttfp_ms": 900.0,
                "completed": 31,
                "failed": 1,
            }
        ),
        encoding="utf-8",
    )
    (raw / "two.json").write_text(
        json.dumps({"mean_audio_rtf": 0.39, "mean_ttft_ms": 330.0, "failed": 0}),
        encoding="utf-8",
    )

    summary = summarize_json_results(run_dir)

    assert summary["metrics"]["mean_audio_rtf"] == [0.4, 0.39]
    assert summary["metrics"]["mean_audio_ttfp_ms"] == [900.0]
    assert summary["metrics"]["mean_ttft_ms"] == [330.0]
    assert summary["metrics"]["failed"] == [1.0, 0.0]
    assert summary["metrics"]["failure_rate"] == [1 / 32]
    assert summary["results"][0]["max_concurrency"] == 1
