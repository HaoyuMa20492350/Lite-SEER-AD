---
title: "Lite-SEER-AD 全文件整合总纲与 SCI 一区发表路线"
subtitle: "从轻量扩散重构到 Feature-First Repair-Aware Anomaly Verification"
author: "综合整理版"
date: "2026-06-20"
lang: zh-CN
CJKmainfont: "Noto Sans CJK SC"
mainfont: "Noto Sans CJK SC"
monofont: "Noto Sans Mono CJK SC"
geometry: margin=1.7cm
fontsize: 9.5pt
linestretch: 1.12
colorlinks: true
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - |
    ```{=latex}
    \usepackage{longtable,booktabs,array}
    \usepackage{xcolor}
    \sloppy
    ```
---

# 文档目的与整合范围

本文将本轮上传的 **11 份材料**统一整合为一份 Lite-SEER-AD 项目总纲、方法蓝图、工程规范、实验方案和 SCI 一区发表路线。材料包括两份深度研究 Markdown、两份对应深度研究 PDF、两份 v2 路线图 MD/PDF、两份 v2 汇编 MD/PDF、`plan.md`、`PLAN1.md`，以及一份更广泛的《当前图像生成研究热点与方向深度分析报告》。

整合遵循三个原则：

1. **不机械重复**：内容相同的 Markdown/PDF 配对只保留一份实质结论，但在附录中逐文件标明其贡献和去向。
2. **按时间和证据强度消解冲突**：旧计划中“仅完成 MVTec”的状态被后续证据包状态取代；旧版 SEV/BRDS 叙事被 HN-SEV/CRV/LC-RDS 取代；主指标补强报告进一步提出 feature-first 架构翻转。
3. **区分事实、计划和建议**：已经跑通的工程证据、文档中规定的接口、尚未独立核验的仓库实现、以及未来研究建议，不混写为同一层级。

> **整合后的最终判断**：Lite-SEER-AD 已经不是纯 idea，而是处在“工程闭环基本完成、机制证据已形成、主指标底盘仍需结构性补强”的论文证据包阶段。当前最合理的最终形态不是 diffusion-first detector，而是 **feature-first detector + diffusion verifier**：强特征分支负责主异常评分与 ROI 提议，轻量扩散分支负责局部验证、反事实修复和预算化推理。

# 执行摘要

Lite-SEER-AD 面向工业 one-class / unsupervised anomaly detection。训练阶段主要使用正常图像，测试阶段要求同时输出：

- 图像级异常分数；
- 像素级异常热图；
- 二值缺陷 mask；
- 局部修复图；
- ROI 级语义置信度、反事实修复得分变化和预算日志。

原始 idea 的主流程是：

```text
轻量扩散正常域重构
-> RGB 残差候选
-> 普通区域语义评估器 SEV
-> 规则式预算感知局部修复 BRDS
-> 检测、定位和修复输出
```

v2 将其替换为：

```text
轻量正常域扩散重构
-> HN-SEV 难负样本语义验证
-> CRV 反事实修复验证
-> LC-RDS 时延约束区域扩散调度
```

主指标补强研究进一步指出，残差优先的主评分入口是 Lite-SEER-AD 弱于 PatchCore、PaDiM、SimpleNet 等强基线的首要原因。因此最终推荐架构是：

```text
Input
-> Frozen feature branch: anomaly prior + ROI proposals
-> HN-SEV: pixel/feature/prototype multi-view verification
-> LC-RDS: choose skip / repair10 / repair25 / native_refine
-> Lightweight diffusion: repair only top-k valuable ROIs
-> Multi-space CRV: pixel + feature + prototype score drop
-> Calibrated late fusion
-> image score / heatmap / mask / repaired image / budget log
```

这一架构的论文问题不再是“扩散能否成为最强纯检测器”，而是：

> **在有限推理预算下，局部扩散修复能否成为异常判断的可量化、可证伪、可调度的反事实证据？**

当前完成度综合判断为：工程闭环约 90%，三大核心机制约 70%，baseline 覆盖约 85%，论文证据包约 75%，可投稿初稿准备度约 60%-65%，但 SCI 一区说服力仍约 45%-55%。决定性变量不是继续堆实验数量，而是能否完成 feature-first 架构翻转、三种机制的稳定性验证，以及诚实而有区分度的论文叙事。

# 1. Idea 演化：从 v1 到 v2，再到推荐的 v2.5

## 1.1 v1：轻量扩散检测与局部修复

v1 的合理性在于：扩散模型只用正常图像学习正常分布，测试时将异常区域“重构回正常”，原图与重构图的差异可以形成异常热图；随后对确认的区域进行局部修复，以避免全图漂移。

v1 已经确定了正确的任务边界：

- 不做开放域文生图；
- 不使用 Stable Diffusion、SD3.5 或完整 MMDiT 作为主干；
- 不以 FID、CLIPScore 等开放域生成指标为主；
- 关注工业异常检测、像素定位、修复质量、时延、显存和 NFE。

v1 的主要局限是：

- RGB residual 易把反光、纹理、边缘错位和重构漂移当作缺陷；
- SEV 容易退化为普通的合成异常分类器；
- 修复图只作为展示，没有进入检测决策；
- BRDS 主要依靠手写规则，算法不可替代性弱；
- 扩散承担全图主评分，既慢又不如强预训练特征稳定。

## 1.2 v2：HN-SEV、CRV、LC-RDS 三条证据链

v2 将论文贡献压缩为三条机制证据链：

| 科学问题 | v2 模块 | 需要证明的结论 |
|---|---|---|
| 高残差是否一定是缺陷 | HN-SEV | 正常 hard negatives 能显著降低反光、纹理和边缘伪异常 |
| 修复是否只是“更好看” | CRV | 修复前后 anomaly score drop 与真实缺陷区域对齐，并改善检测/定位 |
| 扩散预算如何分配 | LC-RDS | 在固定 latency/NFE 下获得更好的精度，或在固定精度下减少成本 |

v2 同时引入两个重要工程策略：

- **分辨率解耦**：低分辨率全局候选 + 原分辨率 ROI 细化；
- **更充分 baseline**：PatchCore、PaDiM、SimpleNet、DRAEM、RD4AD、UniAD、DiffusionAD、DDAD，时间允许再加入 FastFlow、InvAD 等。

## 1.3 v2.5：Feature-First Detector + Diffusion Verifier

主指标补强报告提出最关键的结构性升级：

> 不再要求扩散残差独立承担主 anomaly score；先用强冻结特征表达正常性，再把扩散用于局部验证、修复和解释。

v2.5 的必要性来自当前证据：Lite-SEER-AD 主指标明显弱于 PatchCore、PaDiM、SimpleNet。这些方法的共同优势不是更大，而是更成熟地在预训练特征空间刻画正常 manifold。单纯提高 UNet 容量、增加采样步数或全图修复，无法修复 anomaly score 的判别底盘。

因此，最终应把 v2 三模块嵌入强主评分入口：

- feature branch 提供主 anomaly prior 和 ROI；
- HN-SEV 验证候选区域是否偏离正常特征流形；
- diffusion 只处理 top-k uncertain/high-value ROI；
- CRV 检查修复后是否回归正常流形；
- LC-RDS 根据预期证据收益与时延分配预算。

# 2. 任务定义、假设与输出契约

## 2.1 任务定义

给定工业图像 $x$，方法输出：

1. 图像级异常分数 $s(x)$；
2. 像素级异常图 $M(x)$；
3. 二值缺陷掩码 $B(x)$；
4. 修复后图像 $x'$；
5. 每个 ROI 的 HN-SEV 置信度；
6. 每个 ROI 的 CRV score drop；
7. 每个 ROI 的调度动作、NFE、时延和修复面积日志。

## 2.2 训练假设

- 主扩散骨架使用正常图像训练；
- HN-SEV 可用合成异常作为 positive，但必须加入 clean normal 与 mined hard negatives；
- 强特征分支原则上冻结或只训练轻量 adapter，以控制算力和防止过拟合；
- LC-RDS 可先由规则调度生成 teacher labels，再训练小型 MLP；
- 真实缺陷通常没有无缺陷 ground truth，因此修复评估必须区分合成缺陷协议和真实缺陷协议。

## 2.3 不应扩大到的边界

- 不将方法包装为开放域生成模型；
- 不把“大扩散主干”当作主要创新；
- 不以全图多步扩散为默认推理；
- 不宣称全面 SOTA，除非最终表格确实支持；
- 不将好看的修复图等同于有效的异常证据。

# 3. 核心科学问题与统一假设

## 3.1 科学问题一：残差中的异常证据是否可靠

像素残差 $|x-r|$ 同时包含真实缺陷、光照变化、边缘位移、纹理差异和扩散重构误差。仅通过阈值或普通分类器无法可靠分离。

统一假设是：

> 真异常不仅在像素层面有 residual，而且在预训练特征、正常原型距离和修复后的回归趋势上共同偏离正常性；伪异常往往 residual 高，但 feature normality 与 repair response 不支持其为真实缺陷。

## 3.2 科学问题二：修复能否成为反事实检测证据

CRV 的核心不是“生成一张正常图”，而是构造局部反事实：

> 如果把候选区域修成正常形态，模型的异常判断是否显著下降？

如果下降发生在像素、特征和原型距离多个空间，并与 GT 区域对齐，则 repair 是可量化证据；如果只有视觉平滑而特征 anomaly score 不下降，说明只是抹平纹理。

## 3.3 科学问题三：什么时候值得使用扩散预算

扩散不应均匀处理所有 ROI。调度目标应从“高置信区域多跑几步”升级为：

> 最大化单位时延可获得的额外异常证据。

这使 LC-RDS 具有清晰的约束优化意义，也直接回应强高效基线的压力。

# 4. 最终推荐方法框架

## 4.1 总体流程

```text
输入图像 x
|
+-> A. Frozen Feature Normality Branch
|     -> multi-scale patch features
|     -> prototype / memory / Gaussian normality score
|     -> feature anomaly prior M_f
|     -> ROI proposals C={c_i}
|
+-> B. Lightweight Diffusion Branch
      -> low-resolution normal reconstruction r
      -> pixel/gradient/feature residual M_r

M_f + M_r
-> candidate ROI aggregation
-> HN-SEV multi-view verification
-> LC-RDS action selection
-> top-k local masked DDIM repair
-> CRV multi-space score drop
-> calibrated late fusion
-> final image score, heatmap, mask, repair and budget logs
```

## 4.2 Feature Normality Branch

推荐实现从低风险到高收益分三档：

1. **Prototype distance**：冻结 backbone，多尺度 patch 特征与类别正常原型比较；
2. **Light memory bank**：PatchCore 风格的少量 coreset 最近邻；
3. **Gaussian density**：PaDiM 风格的位置/patch 分布建模。

主目标不是复制完整 baseline，而是为 Lite-SEER-AD 提供可靠的 anomaly prior 和 ROI proposal。第一轮最小实验应比较：

```text
residual only
feature only
feature + HN-SEV
feature + HN-SEV + CRV
```

若 feature-only 已显著高于 residual-only，说明主瓶颈得到验证；后续研究重点转向“扩散验证带来的增量”，而不是继续修补像素残差。

## 4.3 轻量正常域扩散重构

推荐保持 Small UNet / UNet-DDPM/DDIM，不使用开放域大模型。其任务是生成正常域重构或局部修复，不承担全图唯一主评分。

建议默认：

- 全局输入：256 调试，384 主实验；
- 全局 reconstruction：1-step 主协议，5-step 做敏感性分析；
- 局部 repair：10/25 steps；
- ROI 数量：top-1 与 top-3；
- 高价值 ROI 可回到 native resolution；
- 局部 mask 使用 dilation + soft blending，避免修复边界断裂。

## 4.4 HN-SEV：难负样本感知多视图验证

### 训练数据

HN-SEV 使用三类样本：

- synthetic positives：Perlin、DTD、CutPaste、类内纹理扰动；
- clean normal negatives：普通正常 patch；
- mined hard negatives：正常图中 residual 高但真实正常的反光、周期纹理、边缘位移和背景漂移区域。

形式化为：

$$
D_{sev}=\{(R_i^{syn},1),(R_i^{-},0),(R_i^{clean},0)\}.
$$

其中 hard negatives 可由正常图上的高残差连通区域 Top-K 产生：

$$
H_i=|x_i-D_{fast}(x_i)|, \quad R_i^{-}=TopK(CC(H_i)).
$$

### 输入

v2 输入：

```text
original patch + reconstruction patch + residual patch + prototype distance
```

v2.5 建议增加：

```text
feature cosine gap
feature anomaly prior
repair uncertainty
optional normal prompt similarity
```

### 正常原型分支

$$
d_i=\min_{p_j\in P_{normal}}\|f(c_i)-p_j\|_2.
$$

HN-SEV 输出：

$$
p_i=S(x_i,r_i,h_i,d_i,g_i),
$$

其中 $g_i$ 可表示 feature cosine gap 或多尺度 anomaly prior。

### 验收

- synthetic-only < synthetic + hard negatives；
- 加 prototype 后 ROI 判别更稳定；
- FPRR 明显下降；
- 至少在反光、周期纹理和边缘场景给出可视证据；
- 不能以降低真实缺陷 recall 为代价。

## 4.5 CRV：多空间反事实修复验证

对候选 ROI $c_i$ 执行局部修复：

$$
x_i'=LocalDDIM(x,m_i,T_i).
$$

基础 score drop：

$$
\Delta_i=A(c_i;x)-A(c_i;x_i').
$$

v2.5 不只看 pixel residual，而同时计算：

- pixel residual drop；
- feature anomaly score drop；
- prototype distance drop。

形成 repair gain：

$$
G_i=w_p\Delta_i^{pixel}+w_f\Delta_i^{feature}+w_n\Delta_i^{proto}.
$$

最终区域分数可写为：

$$
s_i=\lambda_f M_f(c_i)+\lambda_r M_r(c_i)+\lambda_s p_i+\lambda_c G_i.
$$

关键评价：

- SDR：Score Drop after Repair；
- RDC：Repair-Detection Consistency；
- score drop 与 GT overlap/IoU 的相关性；
- repair 后正常原型距离是否下降；
- background preservation 是否保持。

如果 CRV 只提升视觉质量但不提升 Pixel AP、AUPRO、Dice 或对齐统计，应降为分析模块，不能硬写成主贡献。

## 4.6 LC-RDS：时延约束区域扩散调度

动作空间：

```text
skip / repair10 / repair25 / native_refine / early_stop
```

输入特征：

```text
ROI area
feature anomaly confidence
HN-SEV confidence
prototype distance
residual peak
boundary complexity
ROI count
current budget
```

v2 目标：

$$
\max_{\phi}\sum_i Gain_i(a_i)-\lambda\sum_i Cost_i(a_i).
$$

v2.5 建议直接学习单位时延预期收益：

$$
U_i(a)=\frac{E[\Delta score_i\mid a]}{E[latency_i\mid a]+\epsilon}.
$$

调度器选择满足总预算的动作组合：

$$
\max_{a_1,\ldots,a_K}\sum_i U_i(a_i),\quad
\text{s.t. }\sum_i latency_i(a_i)\le B.
$$

实施顺序：

1. fixed-10 / fixed-25；
2. rule BRDS；
3. rule teacher -> MLP scheduler；
4. expected-utility scheduler；
5. 报告 latency-NFE-accuracy Pareto，而不是只报告 FPS。

若 learned scheduler 不稳定，主论文使用 rule 或 semi-learned 版本，learned 结果放附录。

## 4.7 分数校准与融合

由于 feature prior、residual、HN-SEV 和 CRV 的尺度不同，应先分别校准，再 late fusion：

- percentile / quantile normalization；
- temperature scaling；
- 正常验证集上的 robust z-score；
- 类别内阈值与全局阈值同时报告。

不建议直接用未经校准的线性加权，因为它会造成跨数据集不稳定和阈值依赖指标波动。

## 4.8 激进高收益扩展：Retrieval-Conditioned Repair

对每个 ROI 从正常库检索 nearest normal patch/prototype，将其作为修复条件或初始化：

```text
suspected ROI
-> retrieve nearest normal patch feature
-> condition local diffusion with normal prototype
-> repair
-> compare repaired feature with retrieved normality
```

该方向将 memory-based normality、conditioned diffusion 和 counterfactual verification 真正统一，可能成为 v2.5 最有新意的扩展。但应在 feature-first 基础版本成立后再做。

# 5. 为什么当前主指标弱，以及如何系统解决

## 5.1 根因诊断

### 根因 A：主异常证据停留在像素空间

PatchCore、PaDiM、SimpleNet 的优势来自稳定的预训练特征 normality 建模。Lite-SEER-AD 若以 RGB residual 为主，HN-SEV 和 CRV 只能在偏弱候选图之后“补救”。

**解决**：feature-first 主评分入口；residual 作为辅助而不是主证据。

### 根因 B：扩散承担过多主检测责任

全图扩散不但慢，还容易重构漂移。即使增加步数，也未必提高判别性。

**解决**：扩散只处理 top-k uncertain/high-value ROI；全局主评分由冻结特征完成。

### 根因 C：HN-SEV 的训练分布不够真实

只用 Perlin/DTD/CutPaste 容易学到合成纹理偏差，真实反光、低对比、边缘伪差覆盖不足。

**解决**：真实正常 hard-negative mining、feature-space input、prototype consistency、类别内 difficulty curriculum。

### 根因 D：CRV 只看像素变化

图像被平滑并不等于缺陷被修复。pixel drop 可能虚高。

**解决**：multi-space CRV，要求 feature score 和 prototype distance 同时回归正常。

### 根因 E：分辨率不足损害小缺陷与边界

低分辨率全图有利于速度，但会损失 tiny defect 和边界。

**解决**：coarse-to-fine；全图 256/384，ROI native-resolution refine。

### 根因 F：多分支分数没有校准

跨数据集、跨类别融合权重固定，可能导致某分支支配最终结果。

**解决**：分支级 calibration、类别无关 robust normalization、权重敏感性实验。

## 5.2 最值得优先实施的三步

1. **Feature-first 最小翻转实验**：最可能直接抬升主表。
2. **Feature-space HN-SEV + multi-space CRV**：最能把方法变成机制论文。
3. **Expected-utility LC-RDS + native ROI refine**：最能巩固低预算与边界质量叙事。

## 5.3 不应优先做的改动

- 更大的扩散主干；
- 更多全图采样步数；
- 默认全图修复；
- 在骨架未稳定前直接扩展到更难数据集；
- 只优化 Image AUROC 而忽略 Pixel AP、边界和阈值依赖指标；
- 只制作漂亮 repair panel 而没有 score-drop 统计。

# 6. 当前工程状态与证据包

## 6.1 当前完成度

| 模块 | 估计完成度 | 当前判断 |
|---|---:|---|
| 工程闭环 | 90% | MVTec15、VisA、MPDD 可运行，输出契约、表格、可视化与 baseline runner 已形成 |
| HN-SEV / CRV / LC-RDS | 约 70% | MVTec15 证据较强，VisA/MPDD 有 mini/gate 证据，但并非所有模块跨数据集稳定成立 |
| Baseline 覆盖 | 约 85% | MVTec15 8 个 baseline；VisA/MPDD 6 个 full-category baseline |
| 论文证据包 | 约 75% | 已有主表、效率表、模块 gate、类别差距、Pareto、failure panel |
| 可投稿初稿 | 60%-65% | 可写 Results/Discussion，但缺 3 seeds、正式结构、图表打磨和强叙事 |
| SCI 一区说服力 | 45%-55% | 当前主指标不支持全面 SOTA，需要架构翻转和机制证据强化 |

## 6.2 覆盖状态

- MVTec15 baseline coverage：`120/120`；
- VisA baseline coverage：`72/72`；
- MPDD baseline coverage：`36/36`；
- VisA mini/gate：schema OK，0 failure；
- MPDD mini/gate：schema OK，0 failure。

旧版 `PLAN1.md` 中“先扩展 VisA/MPDD”的内容应视为历史阶段计划；后续证据显示这两套数据已完成更多覆盖。该计划仍保留其验收标准、输出规范和论文组织价值。

## 6.3 已有 paper package

```text
table_main_cross_dataset.csv
table_efficiency_cross_dataset.csv
table_mean_by_dataset_method.csv
table_category_deltas.csv
table_module_gates.csv
fig_pareto_pixel_ap_fps.png
fig_qualitative_failure_panel.png
```

还需将其升级为审稿人一眼能读懂的贡献图：

1. HN-SEV：正常高残差误报被压制的 before/after；
2. CRV：score drop 与 GT overlap 的散点/箱线图；
3. LC-RDS：fixed10/fixed25/rule/learned 的 Pareto；
4. feature-first 翻转：residual-only、feature-only、feature+HN-SEV、full 的对比图。

## 6.4 证据边界

- 文档已明确 I/O、目录、配置、日志和实验协议；
- 部分网络级超参数，如 UNet channel multipliers、attention resolution、beta schedule、optimizer、EMA、batch size、mask dilation 等仍需在最终仓库和附录中固化；
- 本整合稿依据上传材料整合，没有把文档中的“建议路径”误写为已经独立核验的代码事实；
- 外部论文性能数字来自深度研究材料，正式投稿前必须重新以原论文和官方代码核验。

# 7. 工程实现总规范

## 7.1 推荐环境

```text
Python 3.10 / 3.11
PyTorch
Diffusers
OpenCV
scikit-learn
NumPy / pandas / tqdm / matplotlib
PyTorch Lightning 或等价训练框架
Anomalib 或自定义 baseline wrapper
```

论文实验必须固定：依赖版本、seed、GPU、git hash、config 和环境导出。Python 3.12 可保留为日常环境，但不建议作为论文主环境。

## 7.2 推荐代码结构

```text
seer_ad_v2/
  configs/
    mvtec.yaml
    visa.yaml
    mpdd.yaml
    mvtec_ad2.yaml
  data/
    mvtec.py
    visa.py
    mpdd.py
    mvtec_ad2.py
    transforms.py
    hard_negative_mining.py
  synthesis/
    perlin.py
    cutpaste.py
    dtd_texture.py
  models/
    feature_branch/
      backbone.py
      prototype_bank.py
      memory_bank.py
      density_score.py
    diffusion/
      unet.py
      ddpm.py
      ddim.py
      reconstruction.py
      local_repair.py
      native_resolution_refiner.py
    hn_sev/
      region_encoder.py
      multiview_fusion.py
      losses.py
    counterfactual/
      repair_verification.py
      score_drop.py
    scheduler/
      rule_brds.py
      lc_rds.py
      utility_predictor.py
    fusion/
      calibration.py
      score_fusion.py
  baselines/
    patchcore.py
    padim.py
    simplenet.py
    draem.py
    rd4ad.py
    uniad.py
    diffusionad.py
    ddad.py
    invad.py
  evaluation/
    detection_metrics.py
    repair_metrics.py
    counterfactual_metrics.py
    efficiency_metrics.py
    pareto.py
  tools/
    visualize.py
    export_tables.py
    export_figures.py
  train_diffusion.py
  build_feature_bank.py
  mine_hard_negatives.py
  train_hn_sev.py
  train_lc_rds.py
  infer.py
  evaluate.py
```

## 7.3 主命令接口

```bash
python train_diffusion.py --config configs/mvtec.yaml
python build_feature_bank.py --config configs/mvtec.yaml
python mine_hard_negatives.py --config configs/mvtec.yaml
python train_hn_sev.py --config configs/mvtec.yaml
python train_lc_rds.py --config configs/mvtec.yaml
python infer.py --config configs/mvtec.yaml --ckpt <path>
python evaluate.py --pred_dir <path> --dataset mvtec
```

## 7.4 配置最小字段

```yaml
dataset:
  name: mvtec
  root: /path/to/MVTec-AD
  categories: all
  image_size: 384
  native_resolution_eval: true

feature_branch:
  backbone: frozen_backbone
  scoring: prototype
  multi_scale: true
  bank_size: 1000

model:
  diffusion_backbone: small_unet
  reconstruction_steps: 1
  local_repair_steps: [10, 25]
  max_regions: 3

hn_sev:
  input_mode: original_recon_residual_feature_proto
  hard_negative_topk: 20
  prototype_bank_size: 1000

crv:
  spaces: [pixel, feature, prototype]
  topk_regions: 3
  default_weight: 0.35

lc_rds:
  enabled: true
  actions: [skip, repair10, repair25, native_refine]
  latency_budget_ms: 200
  objective: expected_gain_per_ms

evaluation:
  metrics: [image_auroc, pixel_auroc, aupro, pixel_ap, f1, iou, dice]
  repair_metrics: [psnr, ssim, lpips, background_psnr, boundary_consistency]
  mechanism_metrics: [fprr, sdr, rdc, score_drop_alignment]
  efficiency_metrics: [latency, fps, gpu_memory, params, flops, nfe, repaired_area_ratio]

seed: 7
device: cuda
```

CRV 主协议可沿用 `0.35`，`0.5` 作为权重敏感性分析。

## 7.5 单图输出

```text
image_id/
  input.png
  feature_prior.npy
  reconstruction.png
  residual_heatmap.npz
  candidate_roi.png
  hn_sev_score.png
  prototype_distance.png
  verified_roi.png
  crv_score_drop.png
  mask.png
  repair.png
  roi_log.jsonl
```

`roi_log.jsonl` 至少记录 bbox、面积比例、feature score、residual、HN-SEV confidence、prototype distance、调度动作、NFE、修复前后分数、SDR 和实际 latency。

## 7.6 单次实验输出

```text
run_dir/
  config.yaml
  git_hash.txt
  environment.txt
  seed.txt
  metrics.csv
  efficiency.csv
  pareto.csv
  scores.csv
  roi_budget.json
  crv_score_drop.npy
  figures/
  qualitative_cases/
```

# 8. 数据集、Baseline 与指标体系

## 8.1 数据集层级

主实验：

- MVTec AD 15 类；
- VisA 12 类；
- MPDD 全类。

补强实验：

- MVTec AD 2：在骨架稳定后做 smoke 和补充表；
- few-shot：8/16/32 normal samples；
- 3 seeds；
- failure taxonomy；
- texture/object 分组；
- 需要时增加更大规模或跨类别异常检测设置。

## 8.2 Baseline 最低集合

```text
PatchCore
PaDiM
SimpleNet
DRAEM
RD4AD
UniAD
DiffusionAD
DDAD
Ours
```

时间允许加入 FastFlow、InvAD 和其他近期扩散异常检测方法。

对比逻辑：

- PatchCore：特征记忆库上界；
- PaDiM：统计 normality 建模；
- SimpleNet：feature-space anomaly synthesis 与效率；
- DRAEM：图像空间合成异常；
- RD4AD/UniAD：重构、蒸馏和统一检测；
- DiffusionAD/DDAD/InvAD：扩散近邻；
- Lite-SEER-AD：验证 repair-aware evidence 与预算化推理的独特价值。

## 8.3 标准检测指标

```text
Image AUROC
Pixel AUROC
AUPRO
Pixel AP
F1
IoU
Dice
Boundary F-score
```

AUROC 不能成为唯一评价，必须强调 Pixel AP、阈值依赖指标和边界质量。

## 8.4 机制指标

- **FPRR**：False Positive Reduction Rate/Region Rate，衡量 HN-SEV 抑制正常高残差误报；
- **SDR**：Score Drop after Repair，衡量 CRV；
- **RDC**：Repair-Detection Consistency；
- **Score-drop alignment**：SDR 与 GT overlap、IoU 或缺陷置信度的相关性；
- **Latency-NFE Pareto Area**；
- **Native-resolution boundary F-score**。

## 8.5 修复质量协议

合成缺陷有正常 ground truth，可报告：PSNR、SSIM、LPIPS、Masked PSNR、Background PSNR、Boundary consistency。

真实缺陷通常没有无缺陷 ground truth，应报告：

- background preservation；
- repair-detection consistency；
- feature/prototype normality return；
- boundary consistency；
- 定性 success/failure cases；
- 必要时人评，但不能用人评代替机制指标。

## 8.6 效率指标

```text
Latency
FPS
GPU memory
Params
FLOPs
NFE
repaired area ratio
local region ratio
```

必须报告真实硬件上的 latency-NFE-accuracy Pareto，并固定 warm-up、batch size、precision、输入分辨率和计时范围。

# 9. 实验矩阵

## 9.1 最小架构翻转实验 - P0

数据：MVTec5。设置：

| 设置 | 目的 |
|---|---|
| residual only | 当前 diffusion-first 下界 |
| feature only | 检验主指标底盘 |
| feature + HN-SEV | 检验难负样本验证增量 |
| feature + HN-SEV + CRV | 检验修复证据增量 |
| full + LC-RDS | 检验预算 Pareto |

这是信息增益最高的实验，应先于大规模重跑。

## 9.2 HN-SEV 消融

```text
no SEV
synthetic-only SEV
+ clean normal
+ mined hard negatives
+ prototype distance
+ feature cosine gap
```

重点输出：FPRR、Pixel AP、AUPRO、误报类型分组、可视化。

## 9.3 CRV 消融

```text
detection only
repair visualization only
pixel CRV
feature CRV
prototype CRV
multi-space CRV
top-1 / top-3 / full-image repair
```

重点输出：SDR、RDC、score-drop vs GT alignment、Background PSNR、Pixel AP/AUPRO。

## 9.4 LC-RDS 消融

```text
skip
fixed10
fixed25
rule BRDS
learned LC-RDS
expected-utility LC-RDS
native refine
```

重点输出：latency、NFE、repaired area ratio、Pixel AP、AUPRO、Pareto area。

## 9.5 分辨率与采样

- 全图：256/384/512；
- ROI：固定 resize vs native resolution；
- reconstruction：1/5 steps；
- repair：10/25/50 steps；
- ROI 数量：top-1/top-3；
- 必须证明更高分辨率和更多步数是否真的带来有效增益，而不是仅增加成本。

## 9.6 3 Seeds 子集稳定性

推荐统一协议：seed `7, 13, 21`。

- MVTec5；
- VisA：`candle, cashew, pcb4, pipe_fryum`；
- MPDD：`metal_plate, connector, tubes`。

每个 seed 导出相同的 `metrics.csv / efficiency.csv / pareto.csv / roi_log.jsonl / qualitative_cases`。论文报告 mean +/- std、类别级 win count 和贡献方向一致性。

## 9.7 Failure Taxonomy

至少覆盖：

- 极小低对比缺陷；
- 镜面反光；
- 周期纹理；
- 边缘重构漂移；
- 多缺陷紧邻；
- 透明/重叠物体；
- 高正常方差；
- 修复边界不连续；
- repair 后 feature 仍异常；
- scheduler 错误跳过关键 ROI。

# 10. 分阶段路线与验收门槛

## 10.1 当前阶段：证据包重构，而非从零实现

由于主数据集和 baseline coverage 已基本形成，当前不应重复“第 0-5 阶段”的所有基础工作。旧计划中的阶段定义仍可作为审计清单，但实际路线应从下面四步开始。

## 10.2 阶段 A：主指标底盘诊断 - 1 至 2 周

任务：

- 完成 feature-first 最小翻转；
- 统一 feature/residual/CRV calibration；
- 在 MVTec5 跑四设置对比；
- 输出类别级提升与失败原因。

门槛：feature-first 相比 residual-only 在至少 Pixel AP/AUPRO/Image AUROC 中显著改善，否则重新检查 backbone、patch resolution、normal bank 和 ROI proposal。

## 10.3 阶段 B：机制升级 - 2 至 4 周

任务：

- HN-SEV 升级到 feature-space；
- CRV 升级到 multi-space；
- 实现 score-drop alignment 统计；
- 完成贡献图。

门槛：

- HN-SEV 跨 seed 降低 FPRR；
- CRV 至少在一个主定位指标稳定正收益；
- SDR 与真实缺陷对齐具有可解释相关性。

## 10.4 阶段 C：预算调度与边界补强 - 2 至 3 周

任务：

- fixed/rule/learned/utility scheduler 对比；
- native-resolution ROI refine；
- 完成 Pareto 曲线和边界分析。

门槛：在相同 latency 下定位更好，或相同定位水平下 NFE/latency 更低。

## 10.5 阶段 D：稳定性与论文收口 - 3 至 5 周

任务：

- 3 seeds 子集；
- 跨数据集统一结果；
- 重导出 paper package；
- 写 Results、Discussion、Limitations；
- 决定 MVTec AD 2 是否值得追加。

门槛：三条机制至少有两条在两个以上数据集方向一致；第三条若不稳定，应降级到附录或 pivot。

# 11. 论文定位与写作蓝图

## 11.1 推荐定位

英文工作标题：

> **Lite-SEER-AD: Feature-First Repair-Aware Diffusion Verification for Industrial Anomaly Detection under Limited Budgets**

中文定位：

> **面向有限预算工业异常检测的特征优先、修复感知扩散验证框架。**

不要将标题和摘要写成“轻量扩散全面超越 SOTA”。更准确的中心命题是：

> 强特征分支负责可靠检测，局部扩散负责可证伪的反事实修复证据，预算调度器决定何时值得购买这份证据。

## 11.2 推荐贡献表述

1. 提出 feature-first repair-aware anomaly verification 框架，将主 anomaly prior 与局部扩散验证解耦；
2. 提出 HN-SEV，利用正常 hard negatives、prototype consistency 和多视图特征抑制残差伪异常；
3. 提出 multi-space CRV，把修复前后的像素、特征和正常原型回归转化为反事实检测证据；
4. 提出 latency-constrained / expected-utility regional scheduler，在有限 NFE 下选择最有信息价值的 ROI 修复动作；
5. 建立检测、修复、解释和效率统一的评估协议，并公开跨 MVTec AD、VisA、MPDD 的证据包与 failure taxonomy。

贡献数量最终可压缩为 3 条，避免过多模块式表述：第一条讲统一框架，第二条合并 HN-SEV+CRV，第三条讲 LC-RDS 和实证协议。

## 11.3 论文结构

1. **Introduction**：承认 memory/feature baselines 强；指出它们缺少局部反事实 repair evidence；提出有限预算下的验证问题。
2. **Related Work**：feature/memory AD、synthetic anomaly、reconstruction/distillation、diffusion AD、counterfactual repair、efficient inference。
3. **Method**：feature prior、HN-SEV、local diffusion repair、multi-space CRV、LC-RDS、fusion。
4. **Experiments**：数据、baseline、公平协议、主表、消融、效率、repair evidence、稳定性。
5. **Discussion**：为什么 residual-first 失败、何时 repair evidence 有效、何时 scheduler 会错。
6. **Limitations**：低对比、小缺陷、反光、跨域、无真实正常修复 GT、额外时延。
7. **Appendix**：每类结果、全部参数、3 seeds、更多可视化和复现细节。

## 11.4 正文图表矩阵

推荐正文核心图表：

- Fig.1：整体 feature-first + diffusion verifier 框架；
- Fig.2：HN-SEV hard-negative 训练与误报压制；
- Fig.3：multi-space CRV 与 score-drop alignment；
- Fig.4：LC-RDS latency-NFE-accuracy Pareto；
- Fig.5：success/failure panel；
- Table 1：跨数据集主表；
- Table 2：效率与 Pareto 摘要；
- Table 3：三大模块消融；
- Table 4：3 seeds 稳定性；
- Appendix：类别级 delta、权重敏感性、repair 指标和完整日志字段。

# 12. SCI 一区投稿门槛与策略

## 12.1 不建议直接投稿的状态

- 仍以 residual-only 为主评分且明显弱于强 baseline；
- 只有 MVTec AD；
- 缺 SimpleNet、DiffusionAD、DDAD；
- HN-SEV 无稳定 FPRR 收益；
- CRV 只改善修复图；
- LC-RDS 只有 FPS，没有 Pareto；
- 无 3 seeds 或统计稳定性；
- 无真实 failure cases；
- 配置、日志和环境不可复现。

## 12.2 可投稿的最低状态

- MVTec AD + VisA + MPDD 主表完整；
- baseline 至少覆盖上述 8 个方法；
- feature-first 翻转证明主指标底盘改善；
- HN-SEV、CRV、LC-RDS 至少两条跨数据集稳定成立；
- 完整效率 Pareto；
- 修复评价区分合成与真实缺陷；
- 3 seeds 子集稳定性；
- 论文诚实说明非全面 SOTA。

## 12.3 更稳妥的一区状态

- 在 DiffusionAD/DDAD 等扩散近邻上有稳定优势；
- 主指标明显缩小与 PatchCore/PaDiM/SimpleNet 的差距；
- HN-SEV 在反光、纹理、边缘类有强可视和统计证据；
- CRV score drop 与 GT overlap 显著相关；
- LC-RDS Pareto 优势清晰；
- MVTec AD 2 或另一困难设置提供外推证据；
- few-shot 和类别级统计支持部署价值；
- 代码、配置、日志和 appendix 达到可复现标准。

具体期刊的最新 JCR/中科院分区会变化，投稿时应根据最新分区与 scope 重新核验。路线的关键不是先选期刊，而是先达到上述证据门槛。

# 13. 风险与 Pivot 策略

## 13.1 HN-SEV 不稳定

原因：hard negatives 质量差、合成-真实 gap、RGB 输入不足。

Pivot：加入 feature residual/prototype distance；改为 feature-space verifier；若仍不稳定，收敛为 hard-negative robust diffusion AD。

## 13.2 CRV 不提升检测

原因：repair 质量不足、ROI 不准、score drop 与真实缺陷无关。

Pivot：只修 top-1 高置信 ROI；使用 retrieval condition；将 CRV 降为 consistency/explanation analysis；让 HN-SEV 成为主贡献。

## 13.3 Learned LC-RDS 不如规则

原因：teacher label 噪声、动作收益估计不准、样本量不足。

Pivot：rule 或 semi-learned scheduler 做主方法；expected-utility predictor 作为附录；保留 Pareto 证据。

## 13.4 仍弱于 PatchCore/SimpleNet

这不自动意味着论文失败，但必须满足：

- 不宣称纯检测 SOTA；
- 证明 repair-aware evidence 是 baseline 不自然提供的；
- 证明低预算局部修复带来可量化判断增益；
- 在扩散近邻和机制指标上形成强优势；
- 用 feature-first 版本尽量缩小主表差距。

## 13.5 MVTec AD 2 或跨数据集不稳定

将困难基准放补充材料；强化 VisA/MPDD 类别分析；按 texture/object、reflective/non-reflective、tiny/large 分组；诚实保留 failure cases。

# 14. 图像生成趋势报告对 Lite-SEER-AD 的跨方向启示

《当前图像生成研究热点与方向深度分析报告》并非工业异常检测专门报告，但它包含若干与 Lite-SEER-AD 高度一致的宏观判断。该文件的关键内容不应混入异常检测基准结论，而应作为方法设计的跨领域启示。

## 14.1 从“大生成器”转向“生成器 + 验证器 + 下游闭环”

2024-2026 的生成研究不再只追求更高视觉质量，而是强调可控、可验证、可迁移和可落地。评测器、奖励模型、审计器逐渐成为一等公民。

对 Lite-SEER-AD 的启示：

- CRV 不应是附属图像输出，而应是 verifier；
- HN-SEV 是局部结果审计器；
- feature-first + diffusion verifier 符合“中等规模模块组合形成完整闭环”的高性价比研究范式。

## 14.2 Few-Step 的核心是质量-效率 Pareto，而非简单减步

图像生成领域的 one-step/few-step 研究强调：减步本身不是贡献，关键是保持对齐、细节、可控性与鲁棒性。

对 Lite-SEER-AD 的启示：

- LC-RDS 不能只说减少 NFE；
- 必须报告“证据增益/毫秒”；
- 全局 1-step 与局部可变步数比统一多步更合理。

## 14.3 不确定性驱动的局部追加计算

报告推荐的可控编辑方向是：只在 uncertain region 追加采样步数，避免破坏全局。

对 Lite-SEER-AD 的启示：

- top-k ROI repair 与 native-resolution refine 是正确方向；
- LC-RDS 应将不确定性、边界复杂度与预期收益纳入动作；
- background preservation 必须成为修复指标。

## 14.4 Retrieval 与 Synthetic Data 的关系

生成趋势报告提醒：synthetic data 并不天然优于 retrieval，真实检索 baseline 可能非常强。

对 Lite-SEER-AD 的启示：

- 仅靠 Perlin/DTD/CutPaste 不足；
- normal prototype/memory 与 hard-negative mining 更关键；
- retrieval-conditioned repair 是比继续增强合成缺陷更有潜力的方向。

## 14.5 评测器的漂移与被利用风险

生成领域关注 evaluator drift 和 reward hacking。

对 Lite-SEER-AD 的启示：

- HN-SEV/CRV 不能只在训练集阈值上优化；
- 需要跨数据集校准、3 seeds、阈值敏感性和 failure taxonomy；
- multi-space CRV 可降低单一像素指标被“平滑修复”利用的风险。

## 14.6 该报告中与本项目关系较弱但仍保留的内容

该报告还系统覆盖视觉自回归、统一理解-生成、文本组合泛化、视频、3D/NVS、个性化、安全审计、合成数据课程和评测 benchmark。它们不是 Lite-SEER-AD 当前论文主线，因此不应扩张项目范围；其共同方法论价值是：**不从零训练更大底模，而是围绕明确任务、旧 backbone、验证器和可复现评测构建闭环。**

# 15. 下一步最短行动序列

## P0 - 立即执行

1. 在 MVTec5 做 feature-first 最小翻转；
2. 固化 feature branch、score calibration 和 ROI proposal；
3. 把 HN-SEV 输入升级为 feature/prototype 多视图；
4. 把 CRV 升级为 pixel/feature/prototype 三空间；
5. 生成 feature-first 对比图和 CRV alignment 图。

## P1 - 论文可信度

1. MVTec5、VisA 子集、MPDD 子集做 seeds 7/13/21；
2. fixed10/fixed25/rule/learned/utility scheduler Pareto；
3. 重导出跨数据集 paper package；
4. 完成 Results、Discussion、Limitations 初稿；
5. 明确哪些贡献进入正文，哪些降级到附录。

## P2 - 一区增强

1. native-resolution ROI refine；
2. retrieval-conditioned repair；
3. MVTec AD 2 smoke；
4. few-shot 8/16/32；
5. 统计显著性和完整复现包。

# 16. 最终结论

Lite-SEER-AD 的原始价值不是“用扩散替代所有工业异常检测方法”，而是把局部修复转化为检测之后的第二层证据。当前主指标弱于 PatchCore、PaDiM、SimpleNet，说明 residual-first 的主评分入口不足，而不是 HN-SEV、CRV、LC-RDS 所研究的问题没有价值。

因此，最合理的最终版本应是：

> **以强预训练特征表示正常性，以 HN-SEV 审查候选异常，以轻量扩散仅修复高价值 ROI，以 multi-space CRV 验证修复是否真正回归正常流形，再由 LC-RDS 在有限时延内选择最有信息价值的动作。**

这条路线同时解决三件事：

- 用 feature-first 提升主指标底盘；
- 用 repair-aware verification 保住方法新颖性；
- 用 expected-utility scheduling 建立部署与效率叙事。

如果 feature-first 翻转、multi-space CRV 和 3 seeds 稳定性成立，Lite-SEER-AD 将从“主指标不占优的扩散检测器”转变为一篇更有辨识度的机制论文：它回答的是 **有限预算下，局部修复能否成为异常判断的反事实证据**。这比继续扩大扩散模型、增加全图步数或追求旧 benchmark 上零点几的 AUROC，更符合当前项目证据和研究趋势。

# 附录 A：逐文件关键内容覆盖表

| 文件 | 文件角色 | 已整合的关键内容 | 主要落入章节 |
|---|---|---|---|
| `deep-research-report.md` | 主指标补强研究 | 主指标落后根因；feature-first 架构翻转；feature-space HN-SEV；multi-space CRV；expected-utility LC-RDS；retrieval-conditioned repair；不建议扩大 UNet/步数 | 1.3、4、5、9、15 |
| `deep-research-report2.md` | 总体深度研究 | idea 演化；证据边界；方法细节；完成度；替换映射；3 seeds；优先级；投稿路线 | 全文骨架，特别是 1、6、10、12、13 |
| `Lite-SEER-AD 深度研究报告.pdf` | 上述深度研究的排版版 | 核对执行摘要、方法、证据表、路线和参考链接；无独立于 MD 的核心方法增量 | 与 `deep-research-report2.md` 合并 |
| `Lite-SEER-AD 主指标补强的深度研究报告.pdf` | 主指标补强排版版 | 核对 residual-first 诊断、feature-first 方案、短期改进和文献依据 | 与 `deep-research-report.md` 合并 |
| `Lite-SEER-AD_v2_SCI一区稳妥发表路线图_更新版.md` | v2 方法路线 | v1->v2 替换；HN-SEV/CRV/LC-RDS 数学形式；baseline、指标、阶段路线、拒稿风险 | 1.2、3、4、8、10、12 |
| 同名 PDF | 路线图排版版 | 核对目录、表格、推荐配置和最终建议；无独立于 MD 的核心信息 | 与同名 MD 合并 |
| `Lite-SEER-AD_v2_上传文件编号整理汇编_更新版.md` | 更新版汇编 | 任务输出；旧新组件映射；分辨率解耦；代码结构；实验计划；投稿策略 | 2、4、7、8、10 |
| 同名 PDF | 汇编排版版 | 核对旧新文件关系、方法框架、代码结构和阶段计划 | 与同名 MD 合并 |
| `plan.md` | 完整工程与论文总计划 | 训练/推理流程；代码树；命令；YAML；输出协议；指标；阶段验收；实验矩阵；风险 pivot；投稿检查表 | 7、8、9、10、12、13 |
| `PLAN1.md` | 当前阶段执行计划 | MVTec 工程闭环；VisA/MPDD 扩展计划；CRV 权重 0.35；paper evidence package；验收标准；非全面 SOTA 叙事 | 6、9、10、11 |
| `当前图像生成研究热点与方向深度分析报告.pdf` | 跨方向趋势 | 生成器+评测器闭环；few-step Pareto；不确定性局部计算；retrieval vs synthetic；evaluator drift；博士早期高性价比研究范式 | 第 14 章 |

# 附录 B：推荐默认协议

```text
Method: Lite-SEER-AD feature-first repair-aware version
Task: industrial anomaly detection, localization and local repair
Global score: frozen feature normality prior
Auxiliary score: pixel/gradient/feature residual
Global resolution: 384; debug 256
Global diffusion: one-step; 5-step sensitivity
Local repair: masked DDIM 10/25 steps
ROI: top-1/top-3; optional native-resolution refine
HN-SEV: synthetic positives + clean normals + mined normal hard negatives
HN-SEV input: original + reconstruction + residual + feature prior + prototype distance
CRV: pixel + feature + prototype score drop
Default CRV weight: 0.35; 0.5 sensitivity
Scheduler: rule baseline -> MLP -> expected gain per ms
Seeds: 7, 13, 21
Main datasets: MVTec AD, VisA, MPDD
Enhancement: MVTec AD 2, few-shot, retrieval-conditioned repair
```

# 附录 C：论文产物清单

```text
tables/
  table_main_cross_dataset.csv
  table_efficiency_cross_dataset.csv
  table_mean_by_dataset_method.csv
  table_category_deltas.csv
  table_module_gates.csv
  table_ablation_feature_first.csv
  table_ablation_hn_sev.csv
  table_ablation_crv.csv
  table_ablation_lc_rds.csv
  table_seed_stability.csv
  table_few_shot.csv
  table_failure_cases_summary.csv

figures/
  fig_framework.png
  fig_feature_first_flip.png
  fig_hn_sev_false_positive_reduction.png
  fig_crv_score_drop_alignment.png
  fig_lc_rds_pareto.png
  fig_qualitative_success_failure.png
  fig_native_roi_boundary.png
```

# 附录 D：投稿前复现检查清单

- [ ] 主线已改写为 repair-aware anomaly verification，而非模块堆叠；
- [ ] feature-first 翻转实验完成；
- [ ] HN-SEV 有 hard-negative 降误报证据；
- [ ] CRV 有 multi-space score-drop 对齐证据；
- [ ] LC-RDS 有 latency-NFE-accuracy Pareto；
- [ ] MVTec AD、VisA、MPDD 结果完整；
- [ ] PatchCore、PaDiM、SimpleNet、DiffusionAD、DDAD 等强 baseline 公平；
- [ ] 三个 seed 的 mean +/- std 完成；
- [ ] 合成与真实缺陷修复协议分开；
- [ ] failure cases 和 limitation 完整；
- [ ] config、git hash、环境、seed 和 ROI 日志已保存；
- [ ] references.bib 已按原论文核验；
- [ ] 附录包含每类结果、参数、阈值和更多可视化。

# 附录 E：材料中反复引用的代表性研究线索

以下条目来自上传的深度研究材料，用于形成 Related Work 检索清单；正式投稿前应以原始论文页面核验题名、作者、年份和最终出版信息。

- PaDiM: arXiv:2011.08785
- PatchCore: arXiv:2106.08265
- DiffusionAD: arXiv:2303.08730
- EfficientAD: arXiv:2303.14535
- SimpleNet: arXiv:2303.15140
- DDAD: arXiv:2305.15956
- CLIP-ADA: arXiv:2403.09493
- InvAD: arXiv:2404.10760
- AAND: arXiv:2405.02068
- Dinomaly: arXiv:2405.14325
- GLASS: arXiv:2407.09359
- AR-Pro / counterfactual repair direction: arXiv:2410.24178
- PBAS: arXiv:2412.17458
- Diffusion anomaly detection surveys: arXiv:2501.11430, arXiv:2506.09368
- MVTec AD 2: arXiv:2503.21622
