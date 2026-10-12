# 10k 完整状态续训

```bash
hf download Fantasyyy/SDUF-Team-Lingbot-vla-2-competition-10k \
  --include 'resume_dcp/*' --include 'manifest.json' --include 'README.md' \
  --local-dir ./models/sduf-10k
python scripts/verify_download.py ./models/sduf-10k --package resume
```

`resume_dcp/global_step_10000` 内的 `model/`、`optimizer/`、`extra_state/` 是一套完整保存状态，不能只下载某张卡的 distcp 文件。此模型为全参数 AdamW，8 rank FSDP2；更换GPU数量或Torch版本前需验证分片加载与rank状态兼容性。

`resume_dcp/lingbotvla_cli.yaml`、`model_assets/` 保存训练配置和模型资产。还需要：允许的 clean 数据与清单、Qwen配置/tokenizer、LingBot基座/训练教师依赖、匹配训练源码与AMD或CUDA环境。这些依赖不是优化器checkpoint的一部分。

准备环境和新输出目录后，在兼容版本中使用 `enable_resume=true` 与 `load_checkpoint_path=<完整10k目录>`。先查看对应版本训练入口的参数格式，不能将这里的字段名未经核对直接作为某版本的CLI参数。

恢复前检查：

1. 训练实际加载的归一化JSON及机器人映射必须与此10k相配；更换归一化方案应建立新的受控实验分支。
2. 确认恢复后计步为10k、AdamW动量状态非重新初始化；从HF权重开始的新训练应另行标注。
3. 原日程为10k、cosine `1e-4→5e-5`。增加 `max_steps` 可能改变已走过的学习率位置，必须检查恢复后的实际LR及scheduler，不直接假设自然接上。
4. 使用micro batch=16、accumulation=2、global batch=256作为历史基线；若卡数改变，总batch重新核对。保留辅助损失等关键参数。
5. 先短跑真实更新并保存独立检查点，确认loss有限、显存稳定、恢复步数与LR正确；再决定续训长度。

此10k的上游归一化统计来源仍须按赛事要求处理，不能仅靠续训或更换推理统计自动消除该问题。模型保留为诊断对照；最终提交前明确数据、统计来源和评测协议。

本仓库不附一条未经完整跨平台验证的生产续训命令，避免把缺少路径/依赖/恢复检查的示例当成直接长训方案。队友反馈环境版本、恢复日志与首段评测后再确定长训配置。
