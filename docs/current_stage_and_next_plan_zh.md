# Lite-SEER-AD 当前阶段与下一步

## 当前结论

截至 2026-06-15，项目已完成 feature-first 主路线、三数据集三
seed 选择实验、无测试 GT 的固定像素阈值评测，以及 7 个外部基线的
MVTec15 全类别统一复现。MVTec AD 2 本地 public/private 流水线也已完成，
但由于官方网页无法登录，它已从正文主实验和投稿验收门槛中移除，仅保留为
未来可选附录资产。

当前三数据集论文证据包已经本地闭环。剩余投稿工作是选择期刊并转换模板、
补齐作者与声明信息。DiffusionAD 的 15 类完整 3000 轮训练属于可选的第八
外部 baseline 补强，不再与 AD2 一起构成硬门槛。其源码、数据资产、严格
runner 和 16 GB 显存适配均已验证；实测完整矩阵约需 `3801 GPU 小时`。

主方法应表述为：

1. 冻结特征先验负责异常检测和定位。
2. 正常图与确定性合成缺陷用于无真实异常标签的候选选择和像素阈值冻结。
3. HN-SEV 提供区域误报过滤证据。
4. LC-RDS 提供局部扩散预算分配证据。
5. CRV 仅用于可视化和事后检查，不作为主检测增益来源。

## 已完成

- MVTec15、VisA、MPDD 共 33 类、seeds `7/13/23`，合计 99 个
  held-out 主运行。
- 三 seed 候选选择一致率 `100%`。
- 新增 `synthetic_normal_fixed_threshold_v1`：
  - 正常像素 FPR 上限 `0.5%`；
  - 可行阈值中最大化合成 Dice；
  - 不读取真实异常标签或掩码；
  - 主 F1/IoU/Dice 使用冻结阈值；
  - 测试 GT 最优值只保留为 `oracle_*` 诊断字段。
- 99/99 held-out 运行已严格重评，33 个类别级阈值全部通过：
  - 最大正常像素 FPR `0.4968%`；
  - `all_fixed_threshold=true`；
  - 真实异常标签/掩码使用标志均为 false。
- MVTec15 图像分数聚合审计已完成：对 11 种 max、top-k、分位数和
  mean 组合，仅用正常图与 seeds `7/13/23` 的确定性合成缺陷冻结每类模式，
  再审计 45 个 held-out 运行。自动选择均值为 `0.9263`，低于当前
  `top5` 的 `0.9278`，因此正式决策为保留 `top5`，不做事后门控调参。
- AUPRO 更新为组件级 PRO、重复 FPR 合并、`FPR=0.3` 边界插值版本。
- baseline 语义已规范：
  - `PatchCore-Local`、`PaDiM-Local`；
  - `SimpleNet-Lite`、`DRAEM-Lite`、`RD4AD-Lite`、`UniAD-Lite`、
    `DiffusionAD-Lite`、`DDAD-Lite`；
  - 243 个历史 run 和 10 张表已补齐 provenance，共 672 行。
- 新增标准延迟协议 `synchronized_batch_latency_v1`：
  batch size 1、50 warmups、200 measurements、CUDA 同步、p50/p95。
- MVTec AD 2：
  - 数据目录 56/56 路径通过；
  - public 8 类 x 3 seeds = 24/24 运行完成；
  - private 与 private-mixed 共 4090 张图完成导出；
  - submission 根目录只含官方要求的两个目录；
  - 官方 checker 全部通过；
  - 原图尺寸归档 `submissions/mvtec_ad2_seed7.tar.gz` 为 `8.40 GiB`；
  - 推荐上传归档
    `submissions/mvtec_ad2_seed7_model256.tar.gz` 为 `341.67 MiB`；
  - 紧凑包含单一顶层目录、4090 TIFF + 4090 PNG，官方 checker 通过；
  - 紧凑包 SHA256：
    `9ae82e41018ad17a59c5f9ae176f4a12355cef9f7bc9bc143e342145deff266b`。
- 官方 baseline 基础设施：
  - 8 个目标方法均已固定作者仓库与 commit；
  - 8/8 源码环境审计通过；
  - PatchCore、PaDiM、UniAD、DRAEM、DDAD、RD4AD、SimpleNet 均完成
    15/15 类；
  - 其中 PaDiM 是维护中的 Anomalib 参考实现，其余 6 个是作者官方实现；
  - 7 个方法共 105/105 个已运行类别均通过 provenance 和预测产物审计；
  - 全部使用与主方法一致的无真实异常 GT 固定阈值协议；
  - DiffusionAD 作者前景资产 2170/2170 通过校验，单批真实 smoke test
    已完整通过；
  - 用于验证恢复路径的部分训练 checkpoint、预测和训练历史已清除，当前
    不保留任何 `paper_eligible_full_training=false` 的部分训练结果；
  - 16 GB 稳定模式为 effective batch 16、micro 4、AMP；实测
    `20.64 秒/有效 batch`，15 类 `663,000` 步约 `3801 GPU 小时`。

## 严格固定阈值结果

以下为 3 个 held-out seed 的运行级均值：

| Dataset | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Fixed Dice | Oracle Dice |
|---|---:|---:|---:|---:|---:|---:|
| MVTec15 | 0.9278 | 0.9785 | 0.9220 | 0.4639 | 0.4778 | 0.5159 |
| VisA | 0.9609 | 0.9865 | 0.8986 | 0.2600 | 0.2222 | 0.3391 |
| MPDD | 0.8198 | 0.9744 | 0.9069 | 0.2762 | 0.2663 | 0.3257 |
| Overall | 0.9202 | 0.9807 | 0.9107 | 0.3556 | 0.3464 | 0.4170 |

固定 Dice 相比 oracle Dice 平均低 `0.0706`。该差值说明旧测试 GT
阈值会带来明显乐观偏差，因此旧 Dice baseline 增益不能继续作为论文主声明。

阈值无关的 AUPRO 与 Pixel AP 配对结果仍有效：

- AUPRO `+0.0584`，95% CI `[+0.0320, +0.0918]`，胜出 `30/33`。
- Pixel AP `+0.0618`，95% CI `[+0.0419, +0.0821]`，胜出 `31/33`。

这些比较对象是同路径本地工程基线，不是官方复现，因此不能用于通用
external SOTA 声明。

## 官方外部基线对照

下表均使用同一指标实现和
`synthetic_normal_fixed_threshold_v1`。PaDiM 为维护中的参考实现，其余为
固定 commit 的作者官方实现。

| Method | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | Fixed Dice |
|---|---:|---:|---:|---:|---:|
| Lite-SEER-AD | 0.9278 | 0.9785 | 0.9220 | 0.4639 | 0.4778 |
| PatchCore-Official | 0.9893 | 0.9821 | 0.9114 | 0.6236 | 0.4896 |
| PaDiM-Anomalib | 0.8903 | 0.9596 | 0.8797 | 0.4001 | 0.3731 |
| UniAD-Official | 0.9764 | 0.9703 | 0.9045 | 0.4481 | 0.4332 |
| DRAEM-Official | 0.9805 | 0.9749 | 0.9279 | 0.6886 | 0.5232 |
| DDAD-Official | 0.9888 | 0.9748 | 0.9085 | 0.6325 | 0.3420 |
| RD4AD-Official | 0.9789 | 0.9780 | 0.9339 | 0.5680 | 0.4901 |
| SimpleNet-Official | 0.9900 | 0.9752 | 0.9049 | 0.5705 | 0.3528 |

Lite-SEER-AD 在五项均值上全面高于 PaDiM 参考实现，并在 Pixel AUROC、
AUPRO、Pixel AP 和 Fixed Dice 上高于 UniAD。它也在 AUPRO 上高于
PatchCore、DDAD 和 SimpleNet，但 DRAEM 与 RD4AD 的 AUPRO 更高，且多数
作者官方方法的 Image AUROC 和 Pixel AP 更强。当前结果因此支持
label-free policy/threshold selection 的独立贡献，不支持总体领先或 SOTA。

15 类配对统计进一步收紧了这个结论：

- 相对 PaDiM，Pixel AUROC `+0.0189`，95% CI
  `[+0.0106, +0.0280]`，Holm 校正 sign-test `p=0.0068`。
- 相对 PaDiM 的 AUPRO、Pixel AP、Fixed Dice bootstrap CI 为正，但
  Holm 校正 sign-test 不显著，不能写成稳健全面领先。
- 相对 PatchCore、UniAD、DDAD、SimpleNet 的 AUPRO 均值虽更高，
  但 95% CI 均跨 0，只能称为点估计优势。
- DRAEM 和 DDAD 的 Pixel AP 优势在 bootstrap 与 Holm 校正后均成立。
- 共同最大失败类别：Image AUROC 为 `grid`，Pixel AP 多为 `pill`，
  AUPRO 多为 `hazelnut`。

## 图像分数聚合审计

预声明的 label-free 选择器覆盖 11 种聚合方式，并严格排除真实异常标签和
掩码。冻结后在 15 类、3 个 held-out split 上审计：

- 当前统一 `top5`：Image AUROC `0.9278`；
- 自动选择：Image AUROC `0.9263`，差值 `-0.0015`；
- 4 类改善、4 类下降、7 类不变；
- 最大改善为 `pill +0.0378`，最大下降为 `screw -0.0586`。

这说明合成域上的聚合增益不能稳定预测真实缺陷增益。实验已完成但不采用；
不能在查看 held-out 结果后再拟合类别切换阈值，否则会形成元层面的测试泄漏。

## 模块结论

- HN-SEV：平均 FPRR `-0.9187`，33/33 类下降。
- LC-RDS：相对 fixed25/rule 平均时延 `-95.04/-85.05 ms`，
  两项均 33/33 类更快。
- LC-RDS 相对 fixed10 仅 `-0.51 ms`，16/33 类更快，不能声明普遍最快。
- CRV：pooled SDR-GT Spearman `-0.1235`，不支持语义去缺陷、
  GT 对齐或主指标增益声明。
- retrieval-conditioned repair 与多尺度分支为负消融，默认关闭。

## MVTec AD 2 可选资产

public 24 个运行的阈值无关均值为：

- Image AUROC `0.7127`
- Pixel AUROC `0.8149`
- AUPRO `0.4712`
- Pixel AP `0.1001`

public 表中的历史 Dice 使用旧 oracle 阈值，只能作诊断，不进入主声明。
private 阈值由 validation normal 的 `mean + 3 * std` 冻结，submission 已通过
官方 checker。由于官方站点无法登录，当前正文不报告 AD2，也不等待 private
服务器结果。现有 public 结果、private 导出和提交包仅作为未来可选附录资产。

## 论文声明边界

可以声明：

- 无真实异常标签/掩码的候选选择与像素阈值冻结；
- 三数据集阈值无关定位增益和 100% 选择一致率；
- MVTec 上相对 PaDiM 的 Pixel AUROC 稳健优势；
- AUPRO 相对 PatchCore、UniAD、DDAD、SimpleNet 的点估计优势，
  同时明确其配对 CI 跨 0；
- 固定阈值下的绝对 Dice/F1/IoU；
- HN-SEV 的误报过滤作用；
- LC-RDS 相对 fixed25/rule 的效率作用。

不能声明：

- 通用外部 SOTA；
- Lite-SEER-AD 总体优于 DRAEM、RD4AD 或全部作者官方方法；
- 本地 `-Lite` runner 等同于原论文官方实现；
- 旧 oracle Dice 增益为主结果；
- CRV 提升冻结 detector 的 AP/Dice；
- CRV 与真实缺陷区域正相关；
- Lite-SEER-AD 是所有方法中普遍最快。

## 主要证据

- 严格固定阈值主表：`tables/strict_fixed_threshold/`
- 自动门控联合汇总：`tables/feature_first_fusion_aggregate_paper_package/`
- 三 seed 选择：`tables/synthetic_gate_fusion_aggregate_{mvtec15,visa,mpdd}/`
- 模块证据：`tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json`
- few-shot：`tables/fewshot_mvtec/`
- AD2 public/private 状态：`tables/mvtec_ad2_feature_first/`
- AD2 submission protocol：
  `submissions/mvtec_ad2_seed7_model256_metadata/submission_protocol.json`
- 七基线统一对照：`tables/external_baseline_comparison/`
- 七基线配对统计：
  `tables/external_baseline_comparison/table_paired_inference.csv`
- 失败类别归因：
  `tables/external_baseline_comparison/table_worst_category_losses.csv`
- 图像分数聚合负结果：
  `tables/image_score_aggregation_mvtec15/analysis.md`
- 冻结主方法失败案例图与索引：
  `tables/failure_case_panel_mvtec15/`
- 投稿与复现清单：
  `docs/submission_reproducibility_checklist.md`
- RD4AD/SimpleNet 统一导入表：
  `tables/official_{rd4ad,simplenet}_mvtec15/`
- 官方 baseline 就绪度：`tables/official_baseline_readiness/`
- DiffusionAD 实测算力计划：`tables/diffusionad_compute_plan/`
- AD2 上传状态：`docs/mvtec_ad2_upload_status_zh.md`
- 论文草稿：`paper/manuscript.md`

## 验收状态

三数据集主论文验收已完成。MVTec AD 2 官方服务器结果不再计入验收矩阵。
官方外部基线矩阵为 `105/120`；缺失的 15 项全部是可选的 DiffusionAD
完整训练。

## 下一步计划

1. 选定目标期刊，将 `paper/manuscript.md` 转为对应 LaTeX/Word 模板，
   补作者、单位、基金、利益冲突和数据/代码可用性声明。
2. 根据目标期刊篇幅整理三数据集正文表格、失败案例与补充材料，不加入
   AD2 private 占位结果。
3. 如审稿策略确实需要第八个外部 baseline，再申请/配置远程并行 GPU。
   16 GB 本机串行估计为 `158.4` 天，不应直接
   启动全矩阵。大显存节点优先直接跑 author batch 16；若仍需 16 GB，执行
   `python tools/run_official_diffusionad.py --categories all --epochs 3000
   --batch-size 16 --micro-batch-size 4 --amp --resume`。不得以 smoke、
   部分训练 checkpoint 或 `DiffusionAD-Lite` 代替正式结果。
4. 下一轮 Image AUROC 改进不再继续事后搜索聚合公式。优先分析 `grid`、
   `capsule`、`screw` 的正常/真实缺陷分布漂移，并在独立数据集或新冻结的
   validation protocol 上预注册选择规则，再回到 MVTec 做一次最终审计。
5. AD2 网页未来恢复且确有附录需要时，再手动上传现有 checker-passed
   压缩包；该动作不阻塞当前投稿，也不得升级为通用 SOTA 声明。
