# 下载与推理

## 只下载 10k 推理包

```bash
hf download Fantasyyy/SDUF-Team-Lingbot-vla-2-competition-10k-fp32 \
  --local-dir ./models/sduf-10k
```

新仓库权重直接位于 `models/sduf-10k/` 根目录。需要全部6个 safetensors 分片、索引和配套配置。用 `scripts/verify_download.py ./models/sduf-10k` 校验已下载文件；根目录格式会要求清单中的完整推理包。上传未完成时不能直接评测。

`configure_runtime.py` 同时支持新根目录格式和原本地 `inference_fp32/hf_ckpt` 格式。新包不含优化器，完整状态续训仍需另一仓库的 `resume_dcp/`。它使用 LingBot 自定义模型实现，并非可随意用任意 Transformers AutoModel 直接运行。

准备 LingBot 源码及对应的 Torch、Transformers、Flash Attention/MoE 等依赖。历史训练源 SHA 为 `951475ae1b1d87553e7dc47c97b53a3d695c0d13`；优先使用其配套部署实现，不随意切换新版架构。使用 AMD 时还要安装 AMD 适配代码；使用 CUDA 时需要对应的 CUDA 依赖。

生成独立本机配置：

```bash
python scripts/configure_runtime.py \
  --bundle ./models/sduf-10k \
  --source-root /path/to/lingbot-vla-v2 \
  --output ./runtime/10k
```

程序输出本机 `lingbotvla_cli.yaml`，指向正确的模型、Qwen config/tokenizer、归一化文件和机器人配置。由于旧部署代码可能硬编码读取源码目录的 `configs/robot_configs/robotwin.yaml`，启动前须比较它与共享的映射文件；若已有定制配置先保留备份，不覆盖其他实验。

在 LingBot 源码目录启动示例：

```bash
export QWEN3VL_PATH=/absolute/path/models/sduf-10k/qwen3vl_config_tokenizer
export LINGBOTVLA_TRAINING_CONFIG=/absolute/path/runtime/10k/lingbotvla_cli.yaml
python -m deploy.lingbot_vla_v2_policy \
  --model_path /absolute/path/runtime/10k/checkpoints/global_step_10000/hf_ckpt \
  --use_bf16 False --use_fp32 True --use_length 50 \
  --use_compile False --port 13400
```

注意：`LINGBOTVLA_TRAINING_CONFIG` 由部分 AMD 部署适配支持，不能假设任意上游版本均读取此变量。脚本建立 `runtime/10k/checkpoints/global_step_10000/hf_ckpt`，内部逐文件链接到已下载的权重，并在 `runtime/10k` 放置配置，兼容历史上按三层父目录寻找配置的部署代码，也兼容逐层查找的版本。

以上是 FP32/50 的加载起点，并非已证明最佳推理策略。BF16/25 作为历史对照需要显式改相应参数，并保持其他条件一致。首次运行先不 compile，再确认显存、输出有限、动作映射、夹爪方向与回合计数。

RoboTwin 任务 clean 用 seen 指令，randomized 用 unseen 指令；不同任务共同使用一份模型与统一策略。使用指定接口进行完整 policy 评测，如实保留每回合 task、setting、seed、instruction、success 与异常。具体 AMD/CUDA 环境和评测入口按匹配版本 README 配置。

FP32文件约25.5GB不等于推理峰值显存，另有输入、激活与缓存；24GB单卡不能据文件大小保证原FP32模式可用。已有FP32/BF16开关也可能影响不同模块，须按实际代码核对。
