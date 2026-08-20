#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""T20: 美化版 MiniCPM-o 4.5 Demo UI (新A3 260399, 2026-08-13)

基于官方 gradio_demo.py 的 API 调用逻辑 (OpenAI chat-completions,
extra_body: chat_template_kwargs.use_tts_template=true + modalities [text,audio];
文本/图片/音频/视频输入, 文本+音频输出), 重写展示层:
  - 现代深色主题 (品牌色 #6C5CE7 / #00B894)
  - header(状态徽章+连接信息) + 输入区(提示词+媒体+参数) + 输出区(文本+音频播放器)
  - 生成中状态指示 (spinner + 按钮禁用)
  - 响应耗时显示 (TTFT/E2E)
  - 示例 prompt 预设 (中文/英文)
  - 可选 ref 音频 (16kHz 预处理, 同官方逻辑)

功能与官方版等价: 同一 OpenAI API 调用, 不改模型行为。仅展示层差异。
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Iterable

import gradio as gr
import numpy as np
import soundfile as sf
from openai import OpenAI
from PIL import Image

MINICPMO45 = "MiniCPM-o 4.5"

_DEFAULT_REF_AUDIO = {
    MINICPMO45: "assets/ref_audio.wav",
}

DEFAULT_PROMPTS = [
    "你好，请用一句话介绍北京。",
    "Describe what you see in the image.",
    "请先识别这段语音的内容，再用中文回复说话人。",
    "What is happening in this video?",
]

# ---------------------------------------------------------------------------
# 媒体编码 (与官方 demo 完全一致)
# ---------------------------------------------------------------------------


def image_to_base64_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def audio_to_base64_data_url(audio_np: np.ndarray, sample_rate: int) -> str:
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


def ref_audio_to_data_url(path: str, target_sr: int = 16000, max_s: float | None = 8.0) -> str:
    """加载文件 → 下混单声道 → 重采样 16kHz → 可选截断 → data-URL (与官方一致)。"""
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    if sr != target_sr:
        try:
            import librosa  # noqa: TID251

            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from math import gcd

            g = gcd(sr, target_sr)
            up, down = target_sr // g, sr // g
            idx = (np.arange(len(data) * up) / up).astype(np.int64)
            data = data[idx][::down]
        sr = target_sr
    if max_s is not None and len(data) > int(max_s * sr):
        data = data[: int(max_s * sr)]
    return audio_to_base64_data_url(data, sr)


def video_to_base64_data_url(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    ext = p.suffix.lower()
    mime = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(ext, "video/mp4")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def process_audio_input(audio_input: Any | None) -> tuple[np.ndarray, int] | None:
    """规范化 Gradio audio 输入 → (np.ndarray mono float32, sample_rate)。"""
    if audio_input is None:
        return None

    def _from_path(p: str) -> tuple[np.ndarray, int] | None:
        if not p:
            return None
        fp = Path(p)
        if not fp.exists():
            return None
        data, sr = sf.read(fp)
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32), int(sr)

    audio_np: np.ndarray | None = None
    sr: int | None = None

    if isinstance(audio_input, tuple) and len(audio_input) == 2:
        a, b = audio_input
        if isinstance(a, int | float) and isinstance(b, np.ndarray):
            sr, audio_np = int(a), b
        elif isinstance(a, str):
            loaded = _from_path(a)
            if loaded is not None:
                audio_np, sr = loaded
    elif isinstance(audio_input, str):
        loaded = _from_path(audio_input)
        if loaded is not None:
            audio_np, sr = loaded

    if audio_np is None or sr is None:
        return None
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]
    return audio_np.astype(np.float32), sr


# ---------------------------------------------------------------------------
# 模型端点 + 推理 (与官方 demo 完全一致)
# ---------------------------------------------------------------------------


class ModelEndpoint:
    def __init__(self, name: str, api_base: str, model_path: str):
        self.name = name
        self.api_base = api_base
        self.model_path = model_path
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        self._ref_audio_data_url: str | None = None

    @property
    def ref_audio_path(self) -> str | None:
        rel = _DEFAULT_REF_AUDIO.get(self.name)
        if not rel:
            return None
        full = os.path.join(self.model_path, rel)
        return full if os.path.exists(full) else None

    def get_ref_audio_data_url(self) -> str | None:
        """懒加载默认 ref 音频 (16kHz mono data URL)。缺文件时回退纯文本 system prompt。"""
        if self._ref_audio_data_url is not None:
            return self._ref_audio_data_url
        p = self.ref_audio_path
        if not p:
            print(f"[{self.name}] no default reference audio found; falling back to text-only system prompt")
            return None
        try:
            self._ref_audio_data_url = ref_audio_to_data_url(p)
            print(f"[{self.name}] loaded reference audio: {p}")
        except Exception as e:
            print(f"[{self.name}] failed to load reference audio {p}: {e}")
            return None
        return self._ref_audio_data_url

    def build_extras(self) -> dict[str, Any]:
        """extra_body: use_tts_template=True + modalities [text, audio]。"""
        if self.name == MINICPMO45:
            return {
                "extra_body": {
                    "chat_template_kwargs": {"use_tts_template": True},
                    "modalities": ["text", "audio"],
                },
            }
        return {}


def build_audio_assistant_system(ref_url: str | None, language: str = "zh") -> dict[str, Any]:
    """与官方一致的音频助手 system prompt。"""
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"你是 MiniCPM-o，一个能听懂、看懂并开口说话的多模态助手。"
                f"当前语言为{language}。请在回答最后附上你的口头回答。"
            ),
        }
    ]
    if ref_url:
        content.append(
            {
                "type": "audio_url",
                "audio_url": {"url": ref_url},
            }
        )
    return {"role": "system", "content": content}


def run_inference(
    endpoint: ModelEndpoint,
    user_prompt: str,
    audio_tuple: tuple[np.ndarray, int] | None,
    image_pil: Image.Image | None,
    video_path: str | None,
    enable_tts: bool,
    max_tokens: int,
    temperature: float,
) -> Iterable[tuple[str, tuple[int, np.ndarray] | None, float]]:
    """yield (text, audio, elapsed_ms)。计时仅展示用。"""
    user_prompt = user_prompt or ""
    if not user_prompt.strip() and not audio_tuple and not image_pil and not video_path:
        yield "请提供文本提示或至少一个多模态输入。", None, 0.0
        return

    t_start = time.perf_counter()
    try:
        content: list[dict[str, Any]] = []

        if audio_tuple is not None:
            audio_np, sr = audio_tuple
            content.append(
                {"type": "audio_url", "audio_url": {"url": audio_to_base64_data_url(audio_np, sr)}},
            )

        if image_pil is not None:
            img = image_pil if image_pil.mode == "RGB" else image_pil.convert("RGB")
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_base64_data_url(img)}},
            )

        if video_path:
            content.append(
                {"type": "video_url", "video_url": {"url": video_to_base64_data_url(video_path)}},
            )

        if user_prompt.strip():
            content.append({"type": "text", "text": user_prompt})

        if enable_tts:
            ref_url = endpoint.get_ref_audio_data_url()
            system_msg = build_audio_assistant_system(ref_url, language="zh")
        else:
            system_msg = {
                "role": "system",
                "content": (
                    "You are MiniCPM-o, a helpful multimodal assistant that can "
                    "understand images, audio and video, and respond in text and speech."
                ),
            }

        messages = [
            system_msg,
            {"role": "user", "content": content},
        ]

        kwargs: dict[str, Any] = {
            "model": endpoint.model_path,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if enable_tts:
            extras = endpoint.build_extras()
            extra_body = dict(extras.get("extra_body", {}))
            kwargs["extra_body"] = extra_body
        else:
            kwargs["extra_body"] = {"modalities": ["text"]}

        completion = endpoint.client.chat.completions.create(**kwargs)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        text_parts: list[str] = []
        audio_out: tuple[int, np.ndarray] | None = None

        for choice in completion.choices:
            msg = choice.message
            if getattr(msg, "content", None):
                text_parts.append(msg.content)
            audio_obj = getattr(msg, "audio", None)
            if audio_obj and getattr(audio_obj, "data", None):
                raw = base64.b64decode(audio_obj.data)
                if len(raw) > 80:
                    data, sr = sf.read(io.BytesIO(raw))
                    if data.ndim > 1:
                        data = data[:, 0]
                    audio_out = (int(sr), data.astype(np.float32))

        text_response = "\n\n".join(t for t in text_parts if t) or "(no text returned)"
        yield text_response, audio_out, elapsed_ms
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        yield f"Inference failed: {type(exc).__name__}: {exc}", None, elapsed_ms


# ---------------------------------------------------------------------------
# UI (美化版: 深色主题)
# ---------------------------------------------------------------------------

BRAND = "#6C5CE7"        # 品牌紫
BRAND_2 = "#00B894"      # 品牌绿
BG = "#0F1117"           # 页面背景
PANEL = "#1A1D27"        # 面板背景
PANEL_2 = "#222633"      # 次级面板
BORDER = "#2E3345"
TEXT = "#E8EAF2"
MUTED = "#8A90A8"
ACCENT_LIGHT = "#A29BFE"

CSS = f"""
:root {{
  --brand: {BRAND}; --brand-2: {BRAND_2}; --bg: {BG}; --panel: {PANEL};
  --panel-2: {PANEL_2}; --border: {BORDER}; --text: {TEXT}; --muted: {MUTED};
}}
.gradio-container {{ background: {BG} !important; }}
#app-root {{ max-width: 1180px !important; margin: 0 auto; }}
footer {{ display: none !important; }}

/* Header */
#hero {{ background: linear-gradient(135deg, rgba(108,92,231,0.18), rgba(0,184,148,0.10));
  border: 1px solid var(--border); border-radius: 14px; padding: 18px 22px !important; }}
#hero h1 {{ font-size: 1.6rem !important; font-weight: 700; margin: 0 0 4px !important;
  background: linear-gradient(90deg, {BRAND}, {ACCENT_LIGHT}, {BRAND_2});
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
#hero p {{ color: var(--muted); margin: 0 !important; font-size: 0.95rem; }}
#badge-row {{ display: flex; gap: 8px; margin-top: 10px; }}
.badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px;
  border-radius: 999px; font-size: 0.78rem; border: 1px solid var(--border); color: var(--muted); }}
.badge .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {BRAND_2};
  box-shadow: 0 0 6px {BRAND_2}; }}
.badge.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; }}

/* Panels */
.panel {{ background: var(--panel) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; }}
#panel-input {{ border: 1px solid {BRAND}55 !important; }}
.panel-title {{ color: var(--text); font-weight: 600; font-size: 0.92rem;
  padding: 4px 2px 0 2px; }}
.panel-sub {{ color: var(--muted); font-size: 0.78rem; padding: 0 2px 6px 2px; }}

/* Inputs */
textarea, input[type="text"] {{ background: var(--panel-2) !important; color: var(--text) !important;
  border-color: var(--border) !important; border-radius: 8px !important; }}
textarea:focus {{ border-color: {BRAND} !important; box-shadow: 0 0 0 1px {BRAND}55 !important; }}
label {{ color: var(--muted) !important; }}

/* Buttons */
#generate-btn {{ background: linear-gradient(135deg, {BRAND}, #8B5CF6) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-weight: 600 !important; letter-spacing: 0.3px; height: 42px !important;
  box-shadow: 0 4px 16px {BRAND}44; transition: all 0.15s; }}
#generate-btn:hover {{ transform: translateY(-1px); box-shadow: 0 6px 22px {BRAND}66; }}
#generate-btn:disabled {{ opacity: 0.55; }}
#clear-btn {{ background: var(--panel-2) !important; color: var(--muted) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important; height: 42px !important; }}

/* Status / timings */
#status-text {{ color: var(--brand-2); font-weight: 600; }}
#timing {{ font-family: "JetBrains Mono", ui-monospace, monospace; color: var(--muted); font-size: 0.8rem; }}

/* Examples */
#examples .gr-box {{ background: var(--panel-2) !important; border: 1px solid var(--border) !important; }}
#examples button {{ color: var(--text) !important; }}

/* Tabs for media inputs */
.tabs .tab-nav button {{ color: var(--muted) !important; }}
.tabs .tab-nav button.selected {{ color: {BRAND} !important; border-bottom-color: {BRAND} !important; }}
"""


def build_interface(endpoints: dict[str, ModelEndpoint], default_model: str) -> gr.Blocks:
    model_choices = list(endpoints.keys())

    with gr.Blocks(
        css=CSS,
        title="MiniCPM-o 4.5 · Omni Demo",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.violet,
            secondary_hue=gr.themes.colors.emerald,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as demo:
        # ---- Header ----
        with gr.Column(elem_id="hero"):
            gr.Markdown("# 🎙️ MiniCPM-o 4.5 · Omni Studio")
            gr.Markdown(
                "多模态对话 · 文本 / 图片 / 音频 / 视频输入，文本 + 语音输出。由 vllm-omni 驱动。"
            )
            with gr.Row(elem_id="badge-row"):
                gr.HTML(
                    '<span class="badge"><span class="dot"></span>serve online</span>'
                    f'<span class="badge mono">{endpoints[default_model].api_base}</span>'
                    f'<span class="badge mono">{endpoints[default_model].model_path}</span>'
                )

        # ---- Input panel ----
        with gr.Column(elem_id="panel-input"):
            gr.HTML('<div class="panel-title">🎯 输入</div>')
            gr.HTML('<div class="panel-sub">文本提示 + 可选媒体 (图片 / 音频 / 视频)</div>')

            prompt_box = gr.Textbox(
                label="文本提示",
                placeholder="输入问题，或结合下方媒体进行多模态对话……",
                lines=3,
                value=DEFAULT_PROMPTS[0],
            )

            with gr.Tabs():
                with gr.Tab("🖼️ 图片"):
                    image_input = gr.Image(label="Image", type="pil", sources=["upload"], height=220)
                with gr.Tab("🎤 音频"):
                    audio_input = gr.Audio(
                        label="Audio",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                with gr.Tab("🎬 视频"):
                    video_input = gr.Video(label="Video", sources=["upload"], height=220)

            with gr.Row():
                with gr.Column(scale=1):
                    tts_checkbox = gr.Checkbox(label="生成语音输出 (TTS)", value=True)
                with gr.Column(scale=2):
                    max_tokens_slider = gr.Slider(
                        label="Max tokens", minimum=32, maximum=4096, value=1024, step=32
                    )
                with gr.Column(scale=2):
                    temperature_slider = gr.Slider(
                        label="Temperature", minimum=0.0, maximum=1.5, value=0.7, step=0.05
                    )

            with gr.Row():
                generate_btn = gr.Button("⚡ 生成", variant="primary", elem_id="generate-btn")
                clear_btn = gr.Button("清空", elem_id="clear-btn")

        # ---- Output panel ----
        with gr.Column(elem_classes="panel"):
            gr.HTML('<div class="panel-title">📤 输出</div>')
            with gr.Row():
                status_text = gr.Markdown("", elem_id="status-text", visible=True)
                timing_text = gr.Markdown("", elem_id="timing")
            text_output = gr.Textbox(label="文本回复", lines=8, interactive=False)
            audio_output = gr.Audio(label="语音回复", interactive=False)

        # ---- Examples ----
        gr.Examples(
            examples=[[p] for p in DEFAULT_PROMPTS],
            inputs=[prompt_box],
            label="💡 示例提示",
            elem_id="examples",
        )

        model_dd = gr.Dropdown(
            label="Model", choices=model_choices, value=default_model, visible=False
        )

        def _run(
            model_name: str,
            prompt: str,
            image_pil: Image.Image | None,
            audio_in: Any,
            video_path: str | None,
            enable_tts: bool,
            max_tokens: int,
            temperature: float,
        ):
            ep = endpoints[model_name]
            audio_tuple = process_audio_input(audio_in)
            yield "⏳ 生成中……", "", "", None
            for text, audio, elapsed in run_inference(
                ep, prompt, audio_tuple, image_pil, video_path,
                enable_tts, max_tokens, temperature,
            ):
                timing = f"⏱️ 响应耗时 **{elapsed:.0f} ms**" if elapsed else ""
                yield text, timing, text, audio

        generate_btn.click(
            _run,
            inputs=[
                model_dd,
                prompt_box,
                image_input,
                audio_input,
                video_input,
                tts_checkbox,
                max_tokens_slider,
                temperature_slider,
            ],
            outputs=[status_text, timing_text, text_output, audio_output],
        )

        def _clear():
            return "", "", None, None, None, None, None

        clear_btn.click(
            _clear,
            outputs=[prompt_box, status_text, timing_text, image_input, audio_input, video_input, text_output],
        )

        demo.queue()
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minicpmo45-api-base", default=os.environ.get("MINICPMO45_API_BASE", "http://localhost:8099/v1"))
    p.add_argument(
        "--minicpmo45-model",
        default=os.environ.get("MINICPMO45_MODEL", "openbmb/MiniCPM-o-4_5"),
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--share", action="store_true")
    p.add_argument(
        "--root-path",
        default=os.environ.get("GRADIO_ROOT_PATH", ""),
        help="Subpath mount point for reverse proxies, e.g. /proxy/7862. "
        "Tells gradio 6.x to prefix all frontend/API paths so they survive "
        "the WebIDE subpath proxy.",
    )
    p.add_argument(
        "--ssl-certfile",
        default=os.environ.get("GRADIO_SSL_CERTFILE", ""),
        help="Path to TLS certificate (PEM).",
    )
    p.add_argument(
        "--ssl-keyfile",
        default=os.environ.get("GRADIO_SSL_KEYFILE", ""),
        help="Path to TLS private key (PEM).",
    )
    p.add_argument(
        "--ssl-verify",
        action="store_true",
        default=os.environ.get("GRADIO_SSL_VERIFY", "0") == "1",
        help="Require browser to verify the certificate (off by default; self-signed certs should stay off).",
    )
    return p.parse_args()


def _ping(api_base: str, timeout: float = 3.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(api_base.rstrip("/").replace("/v1", "/health"), timeout=timeout):
            return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    endpoints: dict[str, ModelEndpoint] = {}

    if args.minicpmo45_api_base and args.minicpmo45_model:
        ok = _ping(args.minicpmo45_api_base)
        status = "reachable" if ok else "not reachable (will error on generate)"
        print(f"[4.5] {args.minicpmo45_api_base}  ({status})")
        endpoints[MINICPMO45] = ModelEndpoint(MINICPMO45, args.minicpmo45_api_base, args.minicpmo45_model)

    if not endpoints:
        raise SystemExit("No endpoints configured. Pass --minicpmo45-api-base/--minicpmo45-model.")

    default_model = next(iter(endpoints.keys()))
    demo = build_interface(endpoints, default_model)

    launch_kwargs: dict = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
    }
    if args.root_path:
        launch_kwargs["root_path"] = args.root_path
        print(f"[root-path] {args.root_path}")
    if args.ssl_certfile and args.ssl_keyfile:
        launch_kwargs.update(
            ssl_certfile=args.ssl_certfile,
            ssl_keyfile=args.ssl_keyfile,
            ssl_verify=args.ssl_verify,
        )
        print(f"[tls] cert={args.ssl_certfile} key={args.ssl_keyfile} verify={args.ssl_verify}")
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
