# DiffusionAD 官方复现状态

## 结论

截至 2026-06-14，DiffusionAD 的作者源码、MVTec AD、DTD 异常纹理和作者发布的
前景掩码均已准备完成，但作者没有发布 MVTec 模型 checkpoint。默认配置要求每类
训练 3000 轮；按作者 `batch size 16 + drop_last=True` 精确计算，15 类共
`663,000` 个优化步。16 GB RTX 4090 Laptop 上的稳定实测吞吐为每个有效
batch `20.64` 秒，合计约 `3801 GPU 小时 / 158.4 GPU 天`。因此不能用本项目
的 `DiffusionAD-Lite` 替代为官方结果，也不能将短跑 checkpoint 写入论文表。

## 已完成准备

- 作者仓库：
  `https://github.com/HuiZhang0812/DiffusionAD`
- 固定 commit：
  `8a64c035bdbce4c438594b30c6a3dea74fea1af6`
- DTD 图片：5640 张。
- 作者 MVTec 前景资产：2170 张，全部通过图片完整性检查。
- 资产总大小：7,469,718 bytes。
- 聚合 SHA256：
  `990a97dc2f516c8938decb1e2307c65546b9266bfd56bf7f71b4a9f029906fbd`
- 9 个物体类的 `DISthresh` 数量与各自 `train/good` 数量完全一致。
- 纹理类共享的 `carpet/thresh` 共 32 张。
- `imgaug==0.4.0` 已安装，官方训练数据类可成功生成样本。
- 单批真实 smoke test 已完成：
  - `bottle`，1 轮，1 个训练 batch，batch size 1；
  - 83 张测试图推理、严格阈值选择和 provenance 全链路通过；
  - 用时约 28 秒，无显存溢出；
  - 产物明确标记 `paper_eligible_full_training=false`，不得进入论文比较。
- 正式训练路径已真实验证：
  - 作者 batch 16、每个 author batch 一次 Adam/scheduler 更新；
  - 16 GB 下 FP32、AMP、激活重算、saved-tensor CPU offload 和 micro 8
    均 OOM；
  - 稳定模式为 CUDA AMP + micro 4 梯度累积；
  - 损失按总样本数与正常样本数加权回 author-batch 均值；
  - 分割网络 BatchNorm 使用 micro 4 统计，必须作为硬件适配偏差披露；
  - 用于验证恢复路径的部分训练 checkpoint、预测和训练历史已清除；
  - 当前不保留任何 paper-ineligible 的部分训练结果。

物化命令：

```powershell
python tools/materialize_diffusionad_foregrounds.py
```

报告：

`third_party/official_baselines/diffusionad/foreground_assets/mvtec/materialization_report.json`

## 官方协议问题

官方 `train.py` 每 50 轮在 MVTec 测试集上计算 Image AUROC 和 Pixel AUROC，
并用测试指标保存 `params-best.pt`。这属于测试集选模，不能直接用于本项目的严格
协议。

正式复现必须：

1. 保留作者网络、数据合成、损失和 3000 轮配置。
2. 固定使用最后一轮 checkpoint，不在训练期间读取测试标签或掩码。
3. 训练完成后仅评估一次。
4. 像素阈值继续使用 `synthetic_normal_fixed_threshold_v1`。
5. 同时报告作者协议与固定终轮协议的差异，不能混合称为同一结果。

## 算力规模

- 重建 UNet：131,654,403 参数。
- 分割网络：28,373,569 参数。
- 合计：160,027,972 参数。
- MVTec15 默认 batch size 16、drop-last、3000 轮。
- 单类优化步数范围：9000 到 72000。
- 15 类合计：`663,000` 步。
- 16 GB 实测：`20.64` 秒/有效 batch，约 `3801.4` GPU 小时。
- 单类估计：`toothbrush 51.6` 小时到 `hazelnut 412.8` 小时。
- 每类完整可恢复 checkpoint 约 `1.79 GiB`，15 类约 `26.8 GiB`。

实测表与并行排程：

`tables/diffusionad_compute_plan/`

在同速 16 GB GPU 上，即使 15 卡一类一卡，最长的 `hazelnut` 仍约
`17.2` 天。更实际的路线是使用能直接容纳 author batch 16 的大显存 GPU，
减少 micro-batch 串行开销，并按类别并行。

因此当前外部官方矩阵中，DiffusionAD 应标记为
`official source, assets, resumable runner, and measured compute plan ready;
full training pending remote compute`，而不是填入本地轻量近似结果、部分训练
checkpoint 或论文表中抄录值。
