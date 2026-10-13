# SDUF Team · LingBot-VLA 2.0 · 10k 阶段共享

本仓库记录第一阶段 AMD 八卡全参数训练的 **global_step_10000**。配置、脚本、逐任务评测结果公开供团队复现与分析。

模型分成两个公开仓库，大文件分批上传，以各仓库文件清单、校验值及实际下载完整性为准：

- [完整状态续训包](https://huggingface.co/Fantasyyy/SDUF-Team-Lingbot-vla-2-competition-10k)：原始 DCP 分片、优化器、训练状态及原配置，约76.62GB。
- [FP32 推理权重包](https://huggingface.co/Fantasyyy/SDUF-Team-Lingbot-vla-2-competition-10k-fp32)：safetensors 和索引放在根目录，含配置、tokenizer、机器人映射和归一化统计，约25.54GB。不含优化器；用于评测或从权重初始化新训练。

上传过程中仓库可访问不等于权重已齐。FP32包需下载全部6个safetensors分片并校验；续训包需保留全部 model、optimizer、extra_state 文件。原合并式本地目录仍保留，不删除源文件。

| 包/目录 | 内容 | 用途 |
|---|---|---|
| 新FP32仓库根目录（本地旧包为 `inference_fp32/`） | 原始 10k 权重保留 FP32 后导出的 safetensors、tokenizer、模型与推理配置 | 加载评测，或从 10k 权重初始化新的训练分支 |
| `resume_dcp/` | `global_step_10000/{model,optimizer,extra_state}`、原训练配置、model_assets | 在兼容环境中恢复完整训练状态 |
| `qwen3vl_config_tokenizer/` | Qwen3-VL-4B-Instruct 的架构与处理器配置，不含另一份 Qwen 权重 | LingBot 加载器的配置依赖 |

完整 DCP 原件为 **76.589 GB / 71.329 GiB**（文件字节合计）；加配套配置与 model_assets 的续训共享目录为 **76.604 GB**。模型含 6,375,907,511 个张量元素；FP32 权重数据为 **25.504 GB / 23.752 GiB**，本次推理目录含配置实际 **25.520 GB**。另有 **15.887 MB** Qwen架构/tokenizer依赖，整体共享文件约 **102.140 GB**。GB 按十进制，GiB 按二进制。目录清单与实际 SHA256 见 HF 的 `manifest.json`。

## 训练记录

- 起点：赛事指定的 [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b)。
- 数据：RoboTwin 2.0，Aloha-AgileX，50 个 clean 任务，每任务 50 条示范；共 2,500 条。原云端元数据审计合计 549,787 帧。
- 训练：10,000 个优化步，8 卡 FSDP2/full shard，全参数 AdamW；每卡 micro batch=16，梯度累积=2，global batch=256；动作 chunk=50。
- 学习率：`1e-4 → 5e-5`，cosine；保存间隔 5,000 步；原训练未启用视觉增强。
- 原配置 `enable_fp32=false`、`enable_mixed_precision=true`。DCP 保存的参数张量全部为 FP32；保存精度与训练计算精度是不同概念。
- 上游训练代码记录：`951475ae1b1d87553e7dc47c97b53a3d695c0d13`。AMD 镜像适配及环境依赖仍需配套，不能只凭此 SHA 保证跨平台恢复。

原训练参数见 `configs/stage1_10k_original.yaml`；原路径留作实验档案，运行时使用自己的目录。机器人映射与归一化统计见 `configs/robot_configs/robotwin.yaml`、`configs/norm_stats/robotwin_original_used_10k.json`。

**归一化记录：**这份 10k 使用的是上游公开 RoboTwin 统计（count=6,062,592），不是由本次 549,787 帧 clean 数据单独计算的统计。主办方回复“从训练数据的统计量来看哈”；团队目前应按训练数据计算统计建立后续合规基线。当前权重保留作历史分析，不在这里宣称其最终提交资格已获认可。评测原 10k 时须使用训练时配套映射，不能仅换推理统计当作已经修复。

## 已有评测结果（诊断）

| 配置 | clean 成功/实际回合 | randomized 成功/实际回合 | 合计 |
|---|---|---|---|
| BF16，use_length=25 | 274/478（57.3%） | 131/435（30.1%） | 405/913（44.4%） |
| FP32，use_length=50 | 242/478（50.6%） | 114/435（26.2%） | 356/913（39.0%） |

各轮完成 100 个任务配置，但各有 38 个配置未实际执行满 10 回合，合计缺 87 回合。统计来自实际 policy 记录；固定打印的 `Final success rate ... /10` 不能作为真实回合数。两轮同时改变了精度与 use_length，不能分离两者的影响，也不是正式 100 回合/任务的比赛成绩。

表格：`results/actual_task_results.csv`、`results/LingBot_10k_BF16_25与FP32_50评测对照.xlsx`。详细说明见 [评测说明](docs/evaluation.md)。历史评测用的是当时导出的权重；本次重新保留 FP32 导出的包已做逐张量校验，但尚未重新跑闭环评测，不能把历史成功率写成新导出包的实测成绩。

## 队友从哪里开始

1. 阅读 [下载与加载](docs/download_and_inference.md)，按用途只下载需要的 HF 子目录。
2. 准备匹配的 LingBot/RoboTwin 环境，使用 `scripts/configure_runtime.py` 生成本机可用路径配置；保留原始配置档案。
3. 先做加载检查与少量同 seed 评测，确认权重完整、机器人配置及归一化实际加载正确。
4. 如需续训，阅读 [续训说明](docs/resume.md)，下载完整 DCP，并检查优化器、步数、学习率与数据映射后再开长训。

本仓库的 `scripts/archive/` 是历史诊断代码，并非已验证的统一官方提交接口；特别是 actual-episode wrapper 改动了停止条件与指令随机数处理，使用前须审查与比赛正式评测协议的差异。

## 来源

- [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)
- [RoboTwin 2.0 数据](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
- [AMD 环境适配](https://github.com/ZiguanWang/Robotwin-radeon-cloud)
- [赛事](https://tianchi.aliyun.com/competition/entrance/532514)

上游代码与模型的使用遵循各自授权条款；复制的上游配置保留来源。数据集和训练教师依赖按原发布渠道获取。
