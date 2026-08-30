# SPDX-License-Identifier: Apache-2.0
"""Exactly-once in-process prewarm for the MiniCPM-o 4.5 NPU pipeline."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.logger import init_logger

from vllm_omni.benchmarks.ultra_timeline import emit_ultra_timeline_event
from vllm_omni.config.minicpmo45_fastpath import MiniCPMO45FastPathConfig

logger = init_logger(__name__)

PrewarmStatus = Literal["disabled", "warming", "ready", "degraded", "failed"]

PREWARM_PROMPTS: tuple[str, ...] = (
    "请用温暖自然的语气介绍杭州西湖清晨的风景",
    "请清晰讲述人工智能如何帮助日常生活与工作",
    "请像新闻主播一样播报今天城市里的温暖故事",
    "请用轻松活泼的声音说明坚持阅读带来的变化",
    "请耐心解释为什么规律作息能够提升学习效率",
    "请用沉稳真诚的语气描述一次难忘的团队合作",
    "请为第一次远行的人送上一段温柔坚定的祝福",
    "请自然介绍中国传统节日里团圆与分享的意义",
    "请用亲切的口吻讲解雨后天空出现彩虹的原因",
    "请像朋友聊天一样分享保持好奇心的重要价值",
    "请用舒缓语调描绘夜晚海边安静闪烁的灯光",
)

_FATAL_NPUGRAPH_MARKERS = (
    "npugraph capture failed",
    "cannot continue after a failed npugraph capture",
)


class PrewarmResponseError(RuntimeError):
    pass


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _stage_engine_args(stage: Any) -> Mapping[str, Any] | None:
    if isinstance(stage, Mapping):
        value = stage.get("engine_args") or stage.get("yaml_engine_args")
    else:
        value = getattr(stage, "engine_args", None) or getattr(
            stage, "yaml_engine_args", None
        )
    return _mapping(value)


def resolve_prewarm_fastpath(
    stage_configs: list[Any] | tuple[Any, ...] | None,
) -> MiniCPMO45FastPathConfig | None:
    """Find the resolved Stage2 fast-path config embedded by stage_config."""
    for stage in stage_configs or ():
        engine_args = _stage_engine_args(stage)
        if engine_args is None:
            continue
        additional = _mapping(engine_args.get("additional_config"))
        raw = _mapping(additional.get("minicpmo45_fastpath")) if additional else None
        if raw is None:
            continue
        config = MiniCPMO45FastPathConfig.from_mapping(raw)
        if config.prewarm or config.prewarm_ready_gate:
            return config
    return None


def _exception_text(error: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts)


def is_fatal_npugraph_error(error: BaseException) -> bool:
    message = _exception_text(error).lower()
    return any(marker in message for marker in _FATAL_NPUGRAPH_MARKERS)


@dataclass(frozen=True)
class PrewarmSnapshot:
    status: PrewarmStatus
    ready_gate: bool
    completed_prompts: int
    error: str | None


class MiniCPMO45PrewarmCoordinator:
    """Run one full-pipeline warmup per FastAPI app instance.

    The coordinator invokes the already-initialized serving object directly;
    it never opens a loopback HTTP connection and is never triggered by
    ``/health``.
    """

    def __init__(
        self,
        *,
        chat_service: Any,
        engine_client: Any,
        model_name: str,
        enabled: bool,
        ready_gate: bool,
    ) -> None:
        self.chat_service = chat_service
        self.engine_client = engine_client
        self.model_name = model_name
        self.enabled = bool(enabled and chat_service is not None)
        self.ready_gate = bool(ready_gate and self.enabled)
        self.status: PrewarmStatus = "disabled" if not self.enabled else "warming"
        self.completed_prompts = 0
        self.error: str | None = None
        self._instance_id = uuid.uuid4().hex
        self._lock = asyncio.Lock()
        self._done = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._attempted = False

    @property
    def timeline_request_id(self) -> str:
        return f"minicpmo45-prewarm-{self._instance_id}"

    def snapshot(self) -> PrewarmSnapshot:
        return PrewarmSnapshot(
            status=self.status,
            ready_gate=self.ready_gate,
            completed_prompts=self.completed_prompts,
            error=self.error,
        )

    def _emit(self, event: str, **details: object) -> None:
        emit_ultra_timeline_event(
            event,
            request_id=self.timeline_request_id,
            stage="prewarm",
            stream="control",
            error=self.error,
            details={
                "status": self.status,
                "completed_prompts": self.completed_prompts,
                "total_prompts": len(PREWARM_PROMPTS),
                **details,
            },
        )

    def start(self) -> asyncio.Task[None] | None:
        if not self.enabled:
            self._done.set()
            return None
        if self._task is None:
            self._task = asyncio.create_task(
                self.run_once(),
                name=f"minicpmo45-prewarm-{self._instance_id[:8]}",
            )
        return self._task

    async def wait(self) -> None:
        await self._done.wait()

    async def shutdown(self) -> None:
        task = self._task
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # The exception has already been classified and exposed through
            # status/error. Awaiting it here consumes the task exception.
            pass

    async def run_once(self) -> None:
        async with self._lock:
            if self._attempted:
                return
            self._attempted = True
            self.status = "warming"
            self._emit("prewarm_begin")
            try:
                for index, prompt in enumerate(PREWARM_PROMPTS):
                    await self._run_prompt(index, prompt)
                    self.completed_prompts = index + 1
                    self._emit("prewarm_prompt_end", prompt_index=index)
            except asyncio.CancelledError:
                self.error = "cancelled"
                self.status = "degraded"
                self._emit("prewarm_cancelled")
                raise
            except Exception as exc:
                self.error = _exception_text(exc)[:2048]
                if is_fatal_npugraph_error(exc):
                    self.status = "failed"
                    self._emit("prewarm_failed", fatal_npugraph=True)
                    logger.exception(
                        "MiniCPM-o 4.5 prewarm hit a fatal NPUGraph capture error"
                    )
                    raise
                self.status = "degraded"
                self._emit("prewarm_degraded", fatal_npugraph=False)
                logger.exception(
                    "MiniCPM-o 4.5 prewarm degraded; opening the service"
                )
            else:
                self.status = "ready"
                self._emit("prewarm_ready")
            finally:
                self._done.set()

    async def _run_prompt(self, index: int, prompt: str) -> None:
        raw_id = f"minicpmo45-prewarm-{self._instance_id}-{index}"
        external_id = f"chatcmpl-{raw_id}"
        request = ChatCompletionRequest(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0,
            seed=0,
            max_tokens=256,
            modalities=["text", "audio"],
            chat_template_kwargs={
                "use_tts_template": True,
                "enable_thinking": False,
            },
            request_id=raw_id,
        )
        self._emit("prewarm_prompt_begin", prompt_index=index)
        try:
            result = await self.chat_service.create_chat_completion(request, None)
            if hasattr(result, "__aiter__"):
                async for chunk in result:
                    text = str(chunk)
                    lowered = text.lower()
                    if '"error"' in lowered or any(
                        marker in lowered for marker in _FATAL_NPUGRAPH_MARKERS
                    ):
                        raise PrewarmResponseError(text[:2048])
            else:
                error = getattr(result, "error", None)
                if error is not None:
                    raise PrewarmResponseError(str(error))
        finally:
            abort = getattr(self.engine_client, "abort", None)
            if callable(abort):
                outcome = abort(external_id)
                if inspect.isawaitable(outcome):
                    await outcome
            abort_audio = getattr(
                self.chat_service,
                "_abort_async_audio_output",
                None,
            )
            if callable(abort_audio):
                abort_audio(external_id)


def build_prewarm_coordinator(
    *,
    stage_configs: list[Any] | tuple[Any, ...] | None,
    chat_service: Any,
    engine_client: Any,
    model_name: str,
) -> MiniCPMO45PrewarmCoordinator:
    config = resolve_prewarm_fastpath(stage_configs)
    return MiniCPMO45PrewarmCoordinator(
        chat_service=chat_service,
        engine_client=engine_client,
        model_name=model_name,
        enabled=bool(config and config.prewarm),
        ready_gate=bool(config and config.prewarm_ready_gate),
    )
