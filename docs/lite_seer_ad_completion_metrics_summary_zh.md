# Lite-SEER-AD 论文完成度与指标对比总结

生成日期：2026-06-26

本文汇总当前仓库中 Lite-SEER-AD 的论文完成度、主实验指标、baseline 对比、模块消融和可声明边界。结论基于当前本地证据包，而不是早期 diffusion-first 历史快照。

## 1. 总体完成度判断

Lite-SEER-AD 当前已经不是 idea 阶段，而是“本地研究证据基本闭环、外部发布与最终投稿信息尚未闭环”的状态。

| 评估口径 | 当前完成度 | 说明 |
|---|---:|---|
| 核心算法、代码、实验、本地论文证据包 | 约 95%-97% | feature-first 主线、三数据集评估、固定阈值、模块审计、baseline 统计均已形成 |
| 仓库自带 P0 完成度矩阵 | 97.58% | `tables/completion_gap_matrix/summary.json` 中的 P0 mean completion |
| SCI 一区投稿准备度 | 约 88%-90% | 已有 manuscript/supplement/FAQ/声明骨架，但缺模板转换和真实投稿元信息 |
| 真正 100% closeout | 未达成 | `final_100_ready=false`，缺跨硬件、公开 release 链接、Zenodo/HF 信息、作者/基金/COI 等 |

当前默认主线为：

```text
Feature-first Lite-SEER-AD
= label-free policy/threshold selection
+ HN-SEV false-positive suppression
+ LC-RDS budgeted local repair scheduling
```

不应再将 Lite-SEER-AD 表述为 diffusion-first detector，也不应宣称 universal SOTA。

## 2. 完成度分项

| 维度 | 状态 | 当前完成度 | 备注 |
|---|---|---:|---|
| 科学定位 | complete | 100% | 已冻结为 feature-first + label-free + selective regional verification |
| 主检测器与 label-free 像素策略 | complete | 100% | 33 类 policy/threshold 均记录不使用真实异常 label/mask |
| 图像级聚合 | complete with limit | 100% | 11 种聚合审计为负结果，保留 `top5`，不做事后切换 |
| HN-SEV | complete with limit | 100% | 可声明误报压制；不能声明 recall-safe verifier |
| LC-RDS | complete | 100% | production 六预算 sweep 完成 |
| CRV | complete with limit | 100% | 已降级为 visualization/post-hoc audit |
| 扩散修复必要性 | complete | 100% | clean-target 对比显示 diffusion 不是必要执行器 |
| 外部 baseline 与统计 | complete with limit | 95% | 7 个 MVTec 官方/参考 baseline 完成；DiffusionAD full 为可选 P2 |
| 效率与部署 | partial | 96% | 缺第二硬件验证，不能写完整 deployment release claim |
| 代码工程 | complete | 100% | `pytest -q` 当前 305 passed |
| 公开复现 | partial | 94% | 本地 manifest/prediction/threshold bundle 就绪，缺 GitHub Release、Zenodo DOI、HF URLs |
| 论文与投稿 | journal gated | 86% | 缺期刊模板转换、作者/单位/基金/利益冲突/公开链接 |

最终 100% 的直接阻塞项：

- `production:cross_hardware`
- `github_release_url`
- `zenodo_doi`
- `hf_model_url`
- `hf_dataset_url`
- `authors_affiliations`
- `funding_conflicts`
- `availability_links`
- `cover_letter`
- release/submission metadata consistency

## 3. Lite-SEER-AD 主指标

主论文 deployable mask 指标采用 `synthetic_normal_fixed_threshold_v1`。`oracle_*` 仅作为测试 GT 阈值上界诊断，不能作为主声明。

| Dataset | Runs | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Fixed Dice | IoU | Normal FPR | Oracle Dice | Oracle Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MVTec AD | 45 | 0.9278 | 0.9785 | 0.9220 | 0.4639 | 0.4778 | 0.3268 | 0.0028 | 0.5159 | 0.0381 |
| VisA | 36 | 0.9609 | 0.9865 | 0.8986 | 0.2600 | 0.2222 | 0.1338 | 0.0026 | 0.3391 | 0.1169 |
| MPDD | 18 | 0.8198 | 0.9744 | 0.9069 | 0.2762 | 0.2663 | 0.1646 | 0.0049 | 0.3257 | 0.0593 |
| Overall | 99 | 0.9202 | 0.9807 | 0.9107 | 0.3556 | 0.3464 | 0.2271 | 0.0031 | 0.4170 | 0.0706 |

固定阈值相比 oracle Dice 平均低 0.0706，说明旧的 test-GT threshold 会显著乐观，不能作为论文主结果。

## 4. MVTec AD 官方/参考 baseline 对比

下表为 MVTec AD 15 类官方/参考 baseline 对比。`Delta` 表示 Lite-SEER-AD 减 baseline；正数代表 Lite-SEER-AD 更高。

| Method | Image AUROC | Delta | Pixel AUROC | Delta | AUPRO | Delta | Pixel AP | Delta | Fixed Dice | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lite-SEER-AD | 0.9278 | - | 0.9785 | - | 0.9220 | - | 0.4639 | - | 0.4778 | - |
| PatchCore-Official | 0.9893 | -0.0615 | 0.9821 | -0.0036 | 0.9114 | +0.0106 | 0.6236 | -0.1597 | 0.4896 | -0.0119 |
| PaDiM-Anomalib | 0.8903 | +0.0375 | 0.9596 | +0.0189 | 0.8797 | +0.0423 | 0.4001 | +0.0638 | 0.3731 | +0.1047 |
| UniAD-Official | 0.9764 | -0.0486 | 0.9703 | +0.0082 | 0.9045 | +0.0174 | 0.4481 | +0.0158 | 0.4332 | +0.0446 |
| DRAEM-Official | 0.9805 | -0.0527 | 0.9749 | +0.0036 | 0.9279 | -0.0059 | 0.6886 | -0.2247 | 0.5232 | -0.0454 |
| DDAD-Official | 0.9888 | -0.0610 | 0.9748 | +0.0038 | 0.9085 | +0.0135 | 0.6325 | -0.1686 | 0.3420 | +0.1357 |
| RD4AD-Official | 0.9789 | -0.0511 | 0.9780 | +0.0006 | 0.9339 | -0.0119 | 0.5680 | -0.1041 | 0.4901 | -0.0123 |
| SimpleNet-Official | 0.9900 | -0.0622 | 0.9752 | +0.0033 | 0.9049 | +0.0171 | 0.5705 | -0.1066 | 0.3528 | +0.1250 |

### 官方 baseline 统计结论

- 相对 PaDiM-Anomalib，Lite-SEER-AD 的 Pixel AUROC 有稳健正优势：mean delta `+0.0189`，95% CI `[+0.0106, +0.0280]`，Holm-adjusted sign-test `p=0.0068`。
- 相对 PaDiM 的 AUPRO、Pixel AP、Fixed Dice 的 bootstrap CI 为正，但 Holm 校正 sign-test 不显著，不能写成稳健全面领先。
- Lite-SEER-AD 对 PatchCore、UniAD、DDAD、SimpleNet 的 AUPRO 是点估计优势，但 95% CI 跨 0，不能写成稳健领先。
- DRAEM 和 DDAD 在 Pixel AP 上显著强于 Lite-SEER-AD。
- 因此当前结果支持 label-free policy/threshold selection 和 strict fixed-threshold behavior，不支持 universal SOTA。

## 5. 三数据集本地工程 control 对比

这一组是 path-aligned/local engineering controls，用于说明 label-free selection 和 feature-first pipeline 的内部贡献，不等同官方 SOTA。

| Scope | Image AUROC Delta | Pixel AUROC Delta | AUPRO Delta | Pixel AP Delta | Dice Delta | 结论 |
|---|---:|---:|---:|---:|---:|---|
| MPDD | +0.0029 | +0.0126 | +0.0420 | +0.0675 | +0.0647 | 定位指标全面正向 |
| MVTec AD | -0.0094 | +0.0075 | +0.0466 | +0.0441 | +0.0320 | Image AUROC 不提升，定位指标提升 |
| VisA | -0.0020 | +0.0244 | +0.0814 | +0.0810 | +0.0857 | 定位指标明显提升 |
| All 33 类 | -0.0044 | +0.0146 | +0.0584 | +0.0618 | +0.0575 | 主要贡献在像素定位，不在 Image AUROC |

All 33 类阈值无关定位增益：

- AUPRO `+0.0584`，95% CI `[+0.0320, +0.0918]`，胜出 `30/33`。
- Pixel AP `+0.0618`，95% CI `[+0.0419, +0.0821]`，胜出 `31/33`。
- Pixel AUROC `+0.0146`，95% CI `[+0.0075, +0.0243]`，胜出 `28/33`。
- Image AUROC `-0.0044`，95% CI 跨 0，不应作为主提升声明。

## 6. 模块级对比

### 6.1 HN-SEV

| Comparison | Image AUROC Delta | Pixel AUROC Delta | AUPRO Delta | Pixel AP Delta | Dice Delta | FPRR Delta |
|---|---:|---:|---:|---:|---:|---:|
| HN-SEV vs feature-only | -0.0049 | +0.0019 | +0.0165 | -0.0129 | -0.0135 | -0.9187 |

HN-SEV 平均 FPRR 下降 `0.9187`，33/33 类下降，因此可以声明 false-positive suppression。

但 HN-SEV 不能声明 recall-safe：

- TP retention：`15.92%`
- ROI recall before HN-SEV：`87.49%`
- ROI recall after HN-SEV：`15.12%`

因此论文中应写成“误报过滤模块”，而不是“稳定保召回 verifier”。

### 6.2 LC-RDS

| Comparison | Image AUROC Delta | Pixel AUROC Delta | AUPRO Delta | Pixel AP Delta | Dice Delta | Latency Delta | NFE Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| utility LC-RDS vs fixed10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.51 ms | 0.00 |
| utility LC-RDS vs fixed25 | -0.0012 | -0.0010 | -0.0061 | -0.0008 | -0.0021 | -95.04 ms | -36.92 |
| utility LC-RDS vs rule BRDS | -0.0006 | -0.0014 | -0.0056 | -0.0005 | -0.0017 | -85.05 ms | -32.94 |

LC-RDS 的合理声明是：

- 相对 fixed25 和 rule BRDS 显著省时。
- 相对 fixed10 只有轻微平均省时，不能宣称普遍最快。
- 它是 budget allocation 贡献，不是全面性能提升模块。

production 六预算 sweep 已完成：

- budgets：`10/25/50/75/100/150 ms`
- budget runs：`198/198`
- missing budget runs：`0`
- max budget violation rate：`0.0009206`
- observed actions：`skip, repair-5, repair-10, repair-25, native-refine`

### 6.3 CRV

| Comparison | Image AUROC Delta | Pixel AUROC Delta | AUPRO Delta | Pixel AP Delta | Dice Delta | Latency Delta | NFE Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| CRV vs HN-SEV | +0.1405 | +0.0058 | +0.0086 | +0.0102 | +0.0130 | +94.92 ms | +24.62 |

虽然 CRV 在若干聚合指标上有正 delta，但 pooled SDR-GT Spearman 为 `-0.1235`，不支持真实缺陷区域正相关。因此 CRV 只能作为 visualization/post-hoc audit，不能作为 semantic repair 或 counterfactual detection 主贡献。

## 7. 扩散修复必要性

clean-target repair executor 对比给出的当前决策是：

```text
decision = diffusion_not_necessary
best_non_diffusion_executor = partial_conv_inpaint
```

| Executor | Latency Mean | SSIM Mean | LPIPS Mean |
|---|---:|---:|---:|
| partial_conv_inpaint | 12.32 ms | 0.9826 | 0.0191 |
| same_protocol_diffusion | 67.94 ms | 0.9378 | 0.1319 |

因此扩散修复不应再被写成不可替代核心。更稳妥的写法是：Lite-SEER-AD 支持 selective regional repair scheduling，而 diffusion 是可替换 executor 之一。

## 8. DiffusionAD 与 MVTec AD 2 状态

DiffusionAD official full reproduction 当前未完成，但不阻塞默认论文完成度：

- 当前官方/参考 baseline readiness：`105/120`
- 缺失 15 项均为 DiffusionAD full 15 类官方配置训练
- 估算成本：约 `3801 GPUh`
- smoke 或 `DiffusionAD-Lite` 不能替代 official full result

MVTec AD 2 是可选附录资产：

- public 24/24 已跑
- private + private-mixed 4090/4090 已导出
- 官方 checker passed
- 因官方网页登录/上传阻塞，不进入当前主论文验收门槛

## 9. 推荐论文声明

可以声明：

- Lite-SEER-AD 是 feature-first、label-free policy/threshold selection 框架。
- 三数据集 33 类、3 seeds、99 held-out runs 完成。
- 主 fixed threshold 不使用真实异常 label/mask。
- 相对本地 path-aligned controls，AUPRO 和 Pixel AP 有稳定正向提升。
- 在 MVTec AD 上，相对 PaDiM-Anomalib 的 Pixel AUROC 有稳健优势。
- HN-SEV 可压制误报。
- LC-RDS 相对 fixed25/rule BRDS 显著减少时延和 NFE。
- CRV、retrieval、multiscale、image-score aggregation 的负结果被保留并如实报告。

不能声明：

- Universal SOTA。
- Lite-SEER-AD 全面优于 PatchCore、DRAEM、RD4AD、SimpleNet 等官方方法。
- Diffusion 是主检测器。
- CRV 与真实缺陷区域正相关。
- CRV 能证明 semantic defect removal。
- smoke/partial DiffusionAD 可作为 official DiffusionAD baseline。
- 旧 oracle Dice 或 test-GT threshold 是 deployable 主结果。

## 10. 最终一句话总结

Lite-SEER-AD 当前最强的论文价值不是“全面 SOTA”，而是：

> 在不使用真实异常标签或掩码进行策略选择的条件下，建立 feature-first 工业异常定位、固定阈值评估、误报过滤和预算化局部修复调度的一套可审计证据包。

当前技术证据包已接近完成；真正阻塞 100% closeout 的是跨硬件部署验证、公开 release/DOI/HF 链接，以及最终投稿元数据。
