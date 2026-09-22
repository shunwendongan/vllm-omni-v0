# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backend choices must survive both deploy projections and engine construction."""

from dataclasses import fields

import pytest
import torch
from transformers import LlamaConfig
from vllm.config import KernelConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_omni.config.omni_config import (
    OmniStageModelConfig,
    VllmOmniConfig,
    VllmOmniDiffusionStageConfig,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.vllm_config import create_diffusion_vllm_config
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pipeline(execution_type):
    return PipelineConfig(
        model_type="backend-test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="model",
                execution_type=execution_type,
                final_output=True,
            ),
        ),
    )


@pytest.fixture
def tiny_ar_model(tmp_path):
    LlamaConfig(
        architectures=["LlamaForCausalLM"],
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
    ).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
@pytest.mark.parametrize("typed", [False, True])
def test_deploy_backends_reach_engine_args(tmp_path, tiny_ar_model, execution_type, typed):
    path = tmp_path / "deploy.yaml"
    attention = (
        "diffusion_attention_backend: torch_sdpa"
        if execution_type == StageExecutionType.DIFFUSION
        else "attention_backend: triton_attn"
    )
    path.write_text(f"stages:\n  - stage_id: 0\n    linear_backend: TORCH\n    moe_backend: TRITON\n    {attention}\n")
    deploy = load_deploy_config(str(path))
    pipeline = _pipeline(execution_type)
    if typed:
        stage = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
        args = build_legacy_engine_args_dict(stage, model="unused")
    assert args["linear_backend"] == "torch"
    assert args["moe_backend"] == "triton"
    if execution_type == StageExecutionType.DIFFUSION:
        kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
        kwargs.pop("model")
        config = OmniDiffusionConfig.from_kwargs(**kwargs)
        assert config.diffusion_attention_config.default.backend == "torch_sdpa"
        final = create_diffusion_vllm_config(torch.device("cpu"), config)
        assert final.kernel_config.linear_backend == "torch"
        assert final.kernel_config.moe_backend == "triton"
    else:
        names = {f.name for f in fields(OmniEngineArgs)}
        kwargs = {k: v for k, v in args.items() if k in names}
        kwargs.update(model=tiny_ar_model, skip_tokenizer_init=True, max_model_len=64, enforce_eager=True)
        final = OmniEngineArgs(**kwargs).create_engine_config()
        assert final.kernel_config.linear_backend == "torch"
        assert final.kernel_config.moe_backend == "triton"
        assert final.attention_config.backend is AttentionBackendEnum.TRITON_ATTN


@pytest.mark.parametrize("config_cls", [OmniStageModelConfig, OmniDiffusionConfig])
def test_kernel_backend_defaults(config_cls):
    config = config_cls()
    assert config.linear_backend == KernelConfig().linear_backend == "auto"
    assert config.moe_backend == KernelConfig().moe_backend == "auto"


@pytest.mark.parametrize("config_cls", [OmniStageModelConfig, OmniDiffusionConfig])
@pytest.mark.parametrize("name", ["linear_backend", "moe_backend"])
@pytest.mark.parametrize("value", ["cutlas", "", 42])
def test_invalid_kernel_backend_rejected(config_cls, name, value):
    with pytest.raises(ValueError, match=name):
        config_cls(**{name: value})


@pytest.mark.parametrize("name", ["linear_backend", "moe_backend"])
def test_upstream_kernel_normalization(name):
    value = "FLASHINFER-CUTLASS"
    expected = getattr(KernelConfig(**{name: value}), name)
    assert getattr(OmniStageModelConfig(**{name: value}), name) == expected
    assert getattr(OmniDiffusionConfig(**{name: value}), name) == expected


def test_diffusion_projection_is_not_overwritten_by_model_defaults():
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        diffusion_config={"linear_backend": "torch", "moe_backend": "triton"},
    )
    args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    assert args["linear_backend"] == "torch"
    assert args["moe_backend"] == "triton"


@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
def test_explicit_backend_wins_over_engine_extras(execution_type):
    extras = {"linear_backend": "cutlass"}
    with pytest.warns(UserWarning, match="linear_backend.*engine_extras"):
        deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, linear_backend="torch", engine_extras=extras)])
    assert extras == {"linear_backend": "cutlass"}
    pipeline = _pipeline(execution_type)
    legacy = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
    typed = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
    assert build_legacy_engine_args_dict(legacy, model="unused")["linear_backend"] == "torch"
    assert build_engine_args_dict_from_omni_stage_config(typed, model="unused")["linear_backend"] == "torch"


def test_unset_backend_does_not_override_engine_extras():
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, engine_extras={"linear_backend": "torch"})])
    pipeline = _pipeline(StageExecutionType.LLM_AR)
    stage = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
    assert build_engine_args_dict_from_omni_stage_config(stage, model="unused")["linear_backend"] == "torch"


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_explicit_model_backend_preserved_for_diffusion(field):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        model_config={field: "triton"},
    )
    assert build_engine_args_dict_from_omni_stage_config(stage, model="unused")[field] == "triton"


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_conflicting_structured_backend_rejected(field):
    stage = VllmOmniDiffusionStageConfig(
        stage_pipeline_config=_pipeline(StageExecutionType.DIFFUSION).stages[0],
        model_config={field: "triton"},
        diffusion_config={field: "cutlass"},
    )
    with pytest.raises(ValueError, match=f"conflicting {field}"):
        build_engine_args_dict_from_omni_stage_config(stage, model="unused")


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("explicit", [None, "auto"])
def test_qwen3_moe_default_and_explicit_auto_unchanged(typed, explicit):
    pipeline = PipelineConfig(
        model_type="qwen3-omni-test",
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        stages=_pipeline(StageExecutionType.LLM_AR).stages,
    )
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, moe_backend=explicit)])
    if typed:
        stage = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs[0]
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0].to_omegaconf()
        args = build_legacy_engine_args_dict(stage, model="unused")
    assert args["moe_backend"] == ("triton" if explicit is None else "auto")


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
@pytest.mark.parametrize("execution_type", [StageExecutionType.LLM_AR, StageExecutionType.DIFFUSION])
def test_cli_overrides_deploy_backend(field, execution_type):
    pipeline = _pipeline(execution_type)
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, **{field: "cutlass"})])
    config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=deploy,
        cli_overrides={f"stage_0_{field}": "AUTO"},
    )
    args = build_engine_args_dict_from_omni_stage_config(config.stage_configs[0], model="unused")
    assert args[field] == "auto"


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_invalid_deploy_kernel_backend_rejected(tmp_path, field):
    path = tmp_path / "invalid.yaml"
    path.write_text(f"stages:\n  - stage_id: 0\n    {field}: not-a-backend\n")
    with pytest.raises(ValueError, match=field):
        load_deploy_config(str(path))


@pytest.mark.parametrize("field", ["linear_backend", "moe_backend"])
def test_explicit_auto_wins_over_engine_extras(field):
    with pytest.warns(UserWarning, match=f"{field}.*engine_extras"):
        stage = StageDeployConfig(stage_id=0, **{field: "auto"}, engine_extras={field: "triton"})
    assert getattr(stage, field) == "auto"
    assert field not in stage.engine_extras


@pytest.mark.parametrize("backend", [None, "auto", "AUTO"])
def test_diffusion_attention_auto_keeps_default(backend):
    config = OmniDiffusionConfig.from_kwargs(diffusion_attention_backend=backend)
    assert config.diffusion_attention_config.default is None


def test_invalid_diffusion_attention_rejected():
    from vllm_omni.platforms.interface import OmniPlatform

    config = OmniDiffusionConfig.from_kwargs(diffusion_attention_backend="not-a-backend")
    with pytest.raises(ValueError, match="backend"):
        OmniPlatform.validate_diffusion_attn_backend(config.diffusion_attention_config.default.backend)


def test_conflicting_diffusion_attention_rejected():
    with pytest.raises(ValueError, match="mutually exclusive"):
        OmniDiffusionConfig.from_kwargs(
            diffusion_attention_backend="TORCH_SDPA",
            diffusion_attention_config={"default": "FLASH_ATTN"},
        )


@pytest.mark.parametrize("backend", [None, "auto", "AUTO"])
def test_ar_attention_auto_reaches_terminal_config(tiny_ar_model, backend):
    config = OmniEngineArgs(
        model=tiny_ar_model,
        worker_cls="auto",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=64,
        attention_backend=backend,
    ).create_engine_config()
    assert config.attention_config.backend is None


def test_ar_invalid_attention_rejected_by_upstream(tiny_ar_model):
    with pytest.raises(ValueError, match="backend"):
        OmniEngineArgs(
            model=tiny_ar_model,
            worker_cls="auto",
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=64,
            attention_backend="not-a-backend",
        ).create_engine_config()


def test_ar_attention_conflict_rejected_by_upstream(tiny_ar_model):
    with pytest.raises(ValueError, match="mutually exclusive"):
        OmniEngineArgs(
            model=tiny_ar_model,
            worker_cls="auto",
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=64,
            attention_backend="TRITON_ATTN",
            attention_config={"backend": "FLASH_ATTN"},
        ).create_engine_config()
