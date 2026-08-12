# MiniCPM-o 4.5 — 昇腾 910C 复现指南(Submission)

## 0. 环境

| 项 | 值 |
|---|---|
| 硬件 | 昇腾 910C 单卡(Atlas A3) |
| 镜像 | quay.io/ascend/vllm-omni:v0.25.0-a3 |
| 依赖 | stepaudio2-minicpmo + step-audio2==1.0.0(--no-deps) + librosa + hyperpyyaml + jiwer + zhon |
| 权重 | /root/models/MiniCPM-o-4_5(本地化,19G) |
| 分支 | optimization-exploration(fork: KuaaMU/vllm-omni) |
| 数据 | /workspace/vllm-omni-data/{daily-omni,videomme} + /root/seed-tts-eval/seedtts_testset |

## 1. 启动服务

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
cd /vllm-workspace/vllm-omni
./examples/minicpm45_ascend/serve.sh
# → vllm serve /root/models/MiniCPM-o-4_5 --omni --port 8091 #   --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml #   --interleave-mm-strings --allowed-local-media-path /workspace/vllm-omni-data
# 等待 /health 200(首次含 FULL cudagraph 图编译,约 5-8 分钟)
```

## 2. 关键配置(deploy YAML)

- `token2wav_float16: true` + `token2wav_n_timesteps: 4`:FP16 + 4 步 CFM
- Stage0 `max_tokens: 32`:Thinker 精简采样
- Stage1 `cudagraph_mode: FULL_DECODE_ONLY`:**整模型图捕获**(关键优化,RTF 0.40→0.35)
- Stage0/2 `cudagraph_mode: PIECEWISE`

## 3. 三大 Benchmark

```bash
export HF_HUB_OFFLINE=1
# Seed-TTS(性能 + WER):32 条并发 1
./examples/minicpm45_ascend/bench_seedtts.sh
# Daily-Omni(精度):1197 条并发 10,minicpm-interleave
./examples/minicpm45_ascend/bench_dailyomni.sh
# Video-MME(精度):2700 条并发 4,minicpm-frames 96 帧
./examples/minicpm45_ascend/bench_videomme.sh
```

## 4. 结果汇总

| 指标 | 成绩 | 官方基线 |
|---|---|---|
| RTF | **0.35** | 0.4423 |
| TTFT | 316 ms | 333.27 ms |
| TTFP | 777 ms | 986.47 ms |
| WER | 1.07% | 1.414% |
| Video-MME | 69.59% | 69.0(准入≥67) |
| Daily-Omni | 78.09% | 79.5(准入≥77.5) |

## 5. Demo

```bash
MINICPMO45_API_BASE=http://localhost:8091/v1 bash examples/online_serving/minicpmo/run_gradio_demo.sh
# 打开 http://<host>:7862,全模态输入 + 文本/音频输出
```

## 6. 优化清单

1. FP16 + 4 步 CFM(DiT 步数 4x 降 + FP16 计算)
2. Stage0 32-token 采样(降首音频延迟)
3. bincount→scatter_add(rep penalty,NPU 兼容)
4. O2:.item() D2H 同步条件化(min_tokens 前跳过)
5. **FULL_DECODE_ONLY cudagraph(Stage1 整模型图捕获,RTF -12.5%,精度零损失)**
