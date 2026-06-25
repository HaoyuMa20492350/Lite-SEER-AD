# LITE-SEER-AD 仓库文件清单

核对日期：2026-06-15。

## 1. 清单范围与总体规模

仓库当前约有 `870,755` 个物理文件、约 `240 GiB` 数据。不能把这些文件
理解为 87 万份独立代码：其中绝大多数是 `runs/` 内重复生成的 NPZ、
JSONL 和 JSON，以及数据集图片和 MVTec AD 2 提交文件。

| 路径 | 文件数 | 大小 | 作用 |
|---|---:|---:|---|
| `runs/` | 800,387 | 103.28 GiB | 2059 个训练、推理、消融和基线运行 |
| `SEER-AD-dataset/` | 43,513 | 42.22 GiB | 五个数据/纹理资源 |
| `submissions/` | 16,366 | 39.74 GiB | MVTec AD 2 展开提交与压缩包 |
| `baselines/` | 812 | 26.83 GiB | 本地 baseline 代码及七方法外部输出 |
| `third_party/` | 4,479 | 19.83 GiB | 八个固定版本的第三方实现和预训练权重 |
| `tables/` | 4,770 | 8.00 GiB | 148 个分析、表格和论文证据包 |
| 其余代码、测试、文档 | 约 400 | 小于 0.1 GiB | 项目本身 |

本清单逐项说明项目维护的代码、配置、文档和测试。对于约 87 万个按相同契约
生成的实验文件，按目录和文件名模式统一解释，不逐张重复列出图片。

## 2. 根目录文件

| 文件 | 作用 |
|---|---|
| `.gitignore` | 排除数据、运行结果、第三方源码、权重、缓存和提交包，避免大文件进入 Git。 |
| `README.md` | 项目入口，说明架构、实验流程、主要命令、产物契约和当前外部基线状态。 |
| `requirements.txt` | 项目的 Python 依赖版本下限。 |
| `pytest.ini` | 指定 `tests/` 为测试目录，并排除大规模产物目录。 |
| `train_diffusion.py` | 训练正常域轻量扩散重建模型。 |
| `train_feature_prior.py` | 建立冻结视觉特征的正常 patch 记忆库/统计先验。 |
| `mine_hard_negatives.py` | 从正常图重建残差中挖掘容易误报的区域。 |
| `train_hn_sev.py` | 训练 HN-SEV 区域语义验证器及原型库。 |
| `train_lc_rds.py` | 训练或保存 LC-RDS 区域修复预算调度器。 |
| `infer.py` | 完整推理入口，组合 feature prior、HN-SEV、局部扩散、CRV 和 LC-RDS。 |
| `evaluate.py` | 从标准 `predictions.npz` 计算检测、定位、阈值和效率指标。 |
| `visualize.py` | 将推理数组转为热图、掩码和案例图。 |

隐藏目录：

- `.git/`：当前项目 Git 历史。
- `.vscode/`：目前为空，预留编辑器配置。
- `.pytest_cache/`：pytest 最近运行缓存，可以删除并自动重建。
- `__pycache__/`：根脚本的 Python 字节码缓存，可以删除并自动重建。

## 3. 配置文件

| 文件 | 作用 |
|---|---|
| `configs/mvtec.yaml` | MVTec AD 15 类主协议：256px、feature-first、top5 图像聚合和合成缺陷策略选择。 |
| `configs/visa.yaml` | VisA 12 类的同协议配置及数据根目录。 |
| `configs/mpdd.yaml` | MPDD 6 类的同协议配置及数据根目录。 |
| `configs/mvtec_ad2.yaml` | MVTec AD 2 八类 public/private 流水线配置；现为可选附录。 |

## 4. 核心包 `seer_ad_v2/`

包与配置：

| 文件 | 作用 |
|---|---|
| `seer_ad_v2/__init__.py` | Python 包标记。 |
| `seer_ad_v2/config.py` | YAML 加载、递归覆盖及类型安全的配置读取。 |

数据层：

| 文件 | 作用 |
|---|---|
| `data/__init__.py` | 数据子包标记。 |
| `data/datasets.py` | MVTec、VisA、MPDD、MVTec AD 2 和 DTD 数据索引及 Dataset 实现。 |
| `data/defect_synthesis.py` | 生成 blob、scratch、spot、patch 等确定性合成缺陷。 |
| `data/hard_negative_mining.py` | 将异常热图转为 ROI，并保存裁剪后的 hard-negative 样本。 |
| `data/mvtec_ad2_discovery.py` | 搜索并检查本地 MVTec AD 2 文件夹或压缩包。 |
| `data/mvtec_ad2_install.py` | 安全解压完整包或分类别 MVTec AD 2 归档。 |
| `data/mvtec_ad2_request.py` | 解析下载申请页面、校验邮箱并提取下载链接。 |

评价层：

| 文件 | 作用 |
|---|---|
| `evaluation/__init__.py` | 评价子包标记。 |
| `evaluation/heatmap_fusion.py` | 对不同来源热图做缩放、正常统计校准和融合。 |
| `evaluation/metrics_counterfactual.py` | 计算修复前后的异常分数下降。 |
| `evaluation/metrics_detection.py` | 计算 Image/Pixel AUROC、Pixel AP、AUPRO、F1、IoU 和 Dice。 |
| `evaluation/metrics_efficiency.py` | CUDA 同步计时和延迟基准测试。 |
| `evaluation/metrics_plan.py` | FPRR、SDR、一致性、Pareto 面积和计划验收指标。 |
| `evaluation/metrics_repair.py` | PSNR、SSIM、背景质量和边界一致性等修复指标。 |
| `evaluation/module_evidence.py` | 生成模块逐项比较和跨类别汇总。 |
| `evaluation/mvtec_ad2_pipeline.py` | 审计 MVTec AD 2 public 运行、模型与提交准备状态。 |
| `evaluation/mvtec_ad2_results.py` | 校验并标准化官方服务器结果及来源元数据。 |
| `evaluation/mvtec_ad2_submission.py` | 验证提交目录、调用官方 checker、计算 SHA256。 |
| `evaluation/pareto.py` | 写出延迟与质量 Pareto 表。 |
| `evaluation/pixel_policy.py` | 像素图校准、后处理和多尺度候选生成。 |
| `evaluation/pixel_threshold_policy.py` | 在正常图与合成缺陷上冻结像素阈值。 |
| `evaluation/prediction_schema.py` | 定义和兼容 `predictions.npz` 中不同热图字段。 |
| `evaluation/repair_quality.py` | 分析 SDR 与真实缺陷、修复质量之间的相关性。 |
| `evaluation/score_aggregation.py` | max、top-k、分位数等图像分数聚合。 |
| `evaluation/synthetic_validation.py` | 计算 label-free 合成验证指标与候选 utility。 |

模型层：

| 文件 | 作用 |
|---|---|
| `models/__init__.py` | 模型子包标记。 |
| `models/seer_ad_v2.py` | 统一构建扩散、HN-SEV 和 LC-RDS 组件。 |
| `models/feature_prior.py` | 冻结 backbone 特征提取、patch bank、PatchCore/PaDiM 风格正常先验。 |
| `models/counterfactual/__init__.py` | CRV 子包标记。 |
| `models/counterfactual/repair_verification.py` | 用修复前后分数差验证 ROI，并融合 verifier 热图。 |
| `models/diffusion/__init__.py` | 扩散子包标记。 |
| `models/diffusion/ddpm.py` | 高斯扩散训练和采样过程。 |
| `models/diffusion/local_refiner.py` | 对选中 ROI 做局部或原分辨率修复。 |
| `models/diffusion/reconstruction.py` | 批量重建、残差图和融合残差图。 |
| `models/diffusion/unet.py` | TinyUNet、残差块和时间步嵌入。 |
| `models/region_verifier/__init__.py` | 区域验证子包标记。 |
| `models/region_verifier/hn_sev.py` | HN-SEV 网络、输入构造、概率和 focal/BCE 损失。 |
| `models/region_verifier/prototype_bank.py` | 正常/困难负样本原型库和手工特征。 |
| `models/scheduler/__init__.py` | 调度器子包标记。 |
| `models/scheduler/lc_rds.py` | 固定、规则、学习式及期望效用区域修复调度器。 |

通用工具：

| 文件 | 作用 |
|---|---|
| `utils/__init__.py` | 工具子包标记。 |
| `utils/image.py` | 图像与掩码读取、tensor/热图转换及保存。 |
| `utils/io.py` | 目录、JSON 和 checkpoint 的读写。 |
| `utils/run.py` | 记录 Git hash、环境和运行元数据。 |
| `utils/seed.py` | 固定 Python、NumPy 和 PyTorch 随机种子。 |

所有 `__pycache__/*.pyc` 都是上述 `.py` 的自动编译缓存，没有独立科研价值。

## 5. Baseline 代码 `baselines/`

| 文件 | 作用 |
|---|---|
| `baselines/__init__.py` | baseline 包标记。 |
| `baselines/README.md` | 区分本地轻量对照与官方/维护实现，并给出运行方法。 |
| `baselines/registry.py` | 定义八个 baseline 的名称、类型、引用和产物要求。 |
| `baselines/run_baseline.py` | 本地 PatchCore、PaDiM 及多个 `*-Lite` 对照的统一运行器。 |
| `baselines/official_sources.json` | 八个外部方法的仓库、固定 commit、引用和执行命令清单。 |
| `baselines/official_sources.py` | 加载来源清单并验证 provenance。 |

`baselines/external_outputs/mvtec15/` 保存七个完成方法各 15 类的标准化结果：

- `metrics.json`：该类别指标。
- `predictions.npz`：图像分数、像素热图、标签、掩码和路径。
- `pixel_threshold_policy.json`：不使用测试异常 GT 的冻结阈值。
- `provenance.json`：源码 commit、环境、命令和协议来源。
- `synthetic_validation_seed*.npz`：三个 seed 的正常图/合成缺陷选择证据。
- RD4AD 额外保留约 1 GiB/类的 `training_checkpoint.pth`，因此其目录占
  `16.11 GiB`，是 baseline 输出中最大的部分。

## 6. 实验工具 `tools/`

审计与诊断：

| 文件 | 作用 |
|---|---|
| `analyze_pixel_policy_gating.py` | 分析候选像素策略的正常图特征与门控可分性。 |
| `annotate_baseline_provenance.py` | 为历史 baseline 结果补充来源字段。 |
| `audit_crv_cases.py` | 审计 CRV 案例和修复前后变化。 |
| `audit_lite_seer_plan.py` | 按研究计划检查必要实验与产物是否齐全。 |
| `audit_mvtec15_baseline_coverage.py` | 检查 MVTec15 baseline 类别覆盖。 |
| `audit_mvtec_ad2_leaderboard.py` | 校验并归档 AD2 leaderboard 快照。 |
| `audit_official_baseline_readiness.py` | 汇总外部 baseline 的源码、运行和论文资格。 |
| `audit_official_source_environments.py` | 检查第三方仓库、LFS 文件和运行环境。 |
| `check_mvtec_ad2_submission.py` | 调用本地和官方规则检查 AD2 提交。 |
| `compare_experiments.py` | 比较两个运行的指标变化。 |
| `discover_mvtec_ad2.py` | 查找和检查 AD2 下载文件。 |

证据和论文导出：

| 文件 | 作用 |
|---|---|
| `export_contribution_chain.py` | 导出主方法到各模块的贡献链。 |
| `export_cross_dataset_module_evidence.py` | 汇总三数据集模块消融。 |
| `export_cross_dataset_repair_quality.py` | 汇总三数据集修复质量与 SDR。 |
| `export_diffusionad_compute_plan.py` | 估算 DiffusionAD 全量训练的步数、时间和存储。 |
| `export_external_baseline_comparison.py` | 生成七外部 baseline 与主方法的比较和统计检验。 |
| `export_failure_case_panel.py` | 选择失败样本并生成论文案例面板。 |
| `export_feature_first_package.py` | 汇总早期 feature-first 跨数据集结果。 |
| `export_feature_first_paper_package.py` | 生成当前 feature-first 主论文证据包。 |
| `export_figures.py` | 从运行结果导出通用图表。 |
| `export_heldout_pixel_policy_package.py` | 汇总 held-out 像素策略选择结果。 |
| `export_heldout_sota_comparison.py` | 对齐样本后比较 held-out 结果与本地强基线。 |
| `export_mvtec15_comparison.py` | 生成 MVTec15 方法比较表。 |
| `export_mvtec_ad2_submission.py` | 生成 AD2 private/private-mixed 提交目录。 |
| `export_official_patchcore_comparison.py` | 生成官方 PatchCore 专项比较。 |
| `export_paper_evidence_summary.py` | 从表格生成论文证据摘要 Markdown。 |
| `export_paper_package.py` | 生成早期 diffusion-first 论文包。 |
| `export_plan_acceptance.py` | 输出当前 19 项验收表与总状态。 |
| `export_repair_visualization.py` | 选择并导出修复过程可视化。 |
| `export_strict_threshold_paper_artifacts.py` | 生成固定阈值主表、图和审计结果。 |
| `export_tables.py` | 从单次运行导出标准表格。 |

协议冻结、导入与数据准备：

| 文件 | 作用 |
|---|---|
| `fetch_official_baseline_sources.py` | 下载并固定外部方法源码。 |
| `freeze_evidence_protocol.py` | 记录代码、文件哈希和命令，冻结证据协议。 |
| `freeze_pixel_threshold_policy.py` | 将一个运行切换为固定像素阈值。 |
| `freeze_selected_pixel_thresholds.py` | 批量冻结选择后的类别阈值。 |
| `import_external_baseline.py` | 将第三方预测转换为项目统一格式。 |
| `import_mvtec_ad2_official_results.py` | 导入 AD2 官方服务器结果。 |
| `index_datasets.py` | 扫描数据集并生成索引。 |
| `install_mvtec_ad2_archive.py` | 安装 AD2 完整包或类别包。 |
| `materialize_ddad_pretrained.py` | 下载并校验 DDAD 预训练资产。 |
| `materialize_diffusionad_foregrounds.py` | 下载并校验 DiffusionAD 作者前景掩码。 |
| `materialize_patchcore_pretrained.py` | 下载并整理 PatchCore 预训练资产。 |
| `request_mvtec_ad2_download.py` | 提交 AD2 下载申请并处理返回链接。 |

候选结果物化：

| 文件 | 作用 |
|---|---|
| `augment_feature_prior_with_retrieval.py` | 给 feature prior 加入正常 patch 检索库。 |
| `materialize_feature_prior_candidate.py` | 只用已有 feature prior 快速生成候选预测。 |
| `materialize_fixed_pixel_policy.py` | 对已有预测应用固定校准和后处理策略。 |
| `materialize_fused_score.py` | 物化多个图像分数来源的融合结果。 |
| `materialize_heatmap_fusion_candidate.py` | 物化热图加权融合候选。 |
| `materialize_multiscale_candidates.py` | 生成预声明的多尺度像素候选。 |
| `materialize_synthetic_normal_validation.py` | 为候选生成正常图和合成缺陷证据。 |
| `materialize_train_normal_pixel_calibration.py` | 生成基于训练正常图的像素校准。 |
| `materialize_tuned_crv.py` | 将已选 CRV 权重应用到运行产物。 |
| `upgrade_prediction_heatmap_schema.py` | 将旧 `predictions.npz` 升级到当前热图字段规范。 |

批量运行器：

| 文件 | 作用 |
|---|---|
| `run_aggregate_synthetic_gate.py` | 在聚合候选池上运行合成缺陷门控。 |
| `run_dataset_mini.py` | 对任意配置执行小规模数据集实验。 |
| `run_feature_first_mvtec5.py` | feature-first 主运行器，支持 MVTec、VisA、MPDD。 |
| `run_feature_rawscore_from_models.py` | 从已有模型重新生成 raw feature 分数。 |
| `run_few_shot_feature_prior.py` | 执行 8/16/32-shot、三 seed 实验。 |
| `run_heatmap_fusion_candidates.py` | 批量生成和评估热图融合候选。 |
| `run_mvtec15_baselines.py` | 批量运行 MVTec15 本地 baseline。 |
| `run_mvtec15_setting.py` | 批量运行一个指定的 MVTec15 设置。 |
| `run_mvtec_ad2_pipeline.py` | 串联 AD2 数据检查、public 运行、private 导出和审计。 |
| `run_mvtec_ad2_public.py` | 执行 AD2 public 八类评估。 |
| `run_mvtec_mini.py` | 早期 MVTec 小规模实验运行器。 |
| `run_next_phase_mvtec5.py` | 早期 diffusion-first 五类门控与消融运行器。 |
| `run_official_ddad.py` | 使用固定作者实现/权重运行 DDAD。 |
| `run_official_diffusionad.py` | 使用固定作者结构运行可恢复 DiffusionAD 全训练。 |
| `run_official_draem.py` | 使用作者实现和 checkpoint 运行 DRAEM。 |
| `run_official_patchcore.py` | 使用作者 PatchCore 实现运行 15 类。 |
| `run_official_rd4ad.py` | 使用作者 RD4AD 实现训练和评估。 |
| `run_official_simplenet.py` | 使用作者 SimpleNet 实现训练和评估。 |
| `run_official_uniad.py` | 使用作者 UniAD 实现和权重评估。 |
| `run_reference_padim.py` | 使用维护中的 Anomalib PaDiM 参考实现。 |
| `run_synthetic_normal_gate.py` | 对三个 seed 批量运行 label-free 候选选择。 |

搜索、选择和汇总：

| 文件 | 作用 |
|---|---|
| `search_crv_weight.py` | 搜索 CRV 融合权重；当前仅用于消融追溯。 |
| `search_image_score_mode.py` | 比较 max、top-k、分位数等图像聚合。 |
| `search_image_score_sources.py` | 比较 residual、feature、fusion 等图像分数来源。 |
| `search_pixel_heatmap_sources.py` | 比较像素评价热图来源。 |
| `search_pixel_postprocess.py` | 比较 Gaussian、high-pass、top-hat、closing 等后处理。 |
| `select_image_score_aggregation.py` | 仅用正常图和合成缺陷冻结图像聚合方式。 |
| `select_pixel_policy_with_normal_gate.py` | 不使用真实异常标签选择像素候选。 |
| `select_pixel_policy_with_val_split.py` | 使用真实异常划分的 oracle/诊断选择器，不用于主声明。 |
| `summarize_evidence.py` | 汇总早期实验报告。 |
| `summarize_heldout_seed_packages.py` | 汇总 held-out 三 seed 稳定性。 |
| `summarize_retrieval_ablation.py` | 汇总检索修复负消融。 |
| `summarize_synthetic_gate_seeds.py` | 汇总 label-free 门控的三 seed 结果。 |

`tools/__pycache__/` 中与这些脚本同名的 `.pyc` 均为可重建缓存。

## 7. 测试 `tests/`

每个测试文件对应同名模块或流程：

| 文件 | 验证内容 |
|---|---|
| `test_baseline_registry.py` | baseline 清单、名称和必需产物。 |
| `test_dataset_sampling.py` | 数据集采样、路径和数量限制。 |
| `test_export_diffusionad_compute_plan.py` | DiffusionAD 算力估算。 |
| `test_export_external_baseline_comparison.py` | 外部 baseline 对齐、统计和表格。 |
| `test_export_failure_case_panel.py` | 失败样本选择与案例图。 |
| `test_export_official_patchcore_comparison.py` | PatchCore 专项比较。 |
| `test_fetch_official_baseline_sources.py` | 第三方源码固定和获取。 |
| `test_heatmap_fusion.py` | 热图尺寸、正常校准和权重融合。 |
| `test_latency_benchmark.py` | 同步延迟协议。 |
| `test_materialize_ddad_pretrained.py` | DDAD 资产下载与校验。 |
| `test_materialize_diffusionad_foregrounds.py` | DiffusionAD 前景资产物化。 |
| `test_module_evidence.py` | 模块比较行和汇总。 |
| `test_mvtec_ad2_discovery.py` | AD2 文件发现。 |
| `test_mvtec_ad2_install.py` | AD2 安装、完整性和路径安全。 |
| `test_mvtec_ad2_pipeline.py` | AD2 全流程状态审计。 |
| `test_mvtec_ad2_request.py` | AD2 申请表和链接解析。 |
| `test_mvtec_ad2_results.py` | 官方结果来源、覆盖和导入。 |
| `test_mvtec_ad2_submission.py` | 提交目录隔离、checker 和压缩。 |
| `test_official_baseline_sources.py` | 八个外部方法的来源和 provenance。 |
| `test_official_source_environment.py` | 第三方环境和 LFS 资产检查。 |
| `test_pixel_threshold_policy.py` | 固定阈值选择和正常 FPR 上限。 |
| `test_prediction_schema.py` | NPZ 热图字段兼容。 |
| `test_repair_quality.py` | 修复质量和相关性统计。 |
| `test_retrieval_scheduler.py` | 检索修复与预算调度。 |
| `test_run_mvtec15_baselines.py` | baseline 批量命令与结果审计。 |
| `test_run_official_ddad.py` | DDAD 官方运行器。 |
| `test_run_official_diffusionad.py` | DiffusionAD checkpoint、micro-batch 和恢复。 |
| `test_run_official_draem.py` | DRAEM 官方运行器。 |
| `test_run_official_patchcore.py` | PatchCore 官方运行器。 |
| `test_run_official_rd4ad.py` | RD4AD 官方运行器。 |
| `test_run_official_simplenet.py` | SimpleNet 官方运行器。 |
| `test_run_official_uniad.py` | UniAD 官方运行器。 |
| `test_run_reference_padim.py` | PaDiM 维护实现。 |
| `test_select_image_score_aggregation.py` | label-free 图像聚合选择。 |
| `test_strict_threshold_paper_artifacts.py` | 固定阈值论文表的完整性和声明边界。 |
| `test_synthetic_policy.py` | 合成验证 utility、稳定性和门控。 |

`tests/__pycache__/` 是测试字节码缓存。

## 8. 文档与论文

`docs/`：

| 文件 | 作用 |
|---|---|
| `current_stage_and_next_plan_zh.md` | 当前完成度、证据、缺口和下一步的权威中文状态。 |
| `diffusionad_reproduction_status_zh.md` | DiffusionAD 资产、协议、算力和未完成状态。 |
| `feature_first_evidence_summary.md` | 历史 feature-first 结果快照，已标记非权威。 |
| `feature_first_execution_record_zh.md` | 当前 label-free 三 seed 执行记录。 |
| `implementation_map.md` | 研究主张到代码、实验和产物的映射。 |
| `literature_matrix.md` | 数据集和方法在论文中的定位。 |
| `mvtec_ad2_leaderboard_template.json` | AD2 leaderboard 归档占位模板。 |
| `mvtec_ad2_official_result_template.json` | AD2 官方结果导入占位模板。 |
| `mvtec_ad2_upload_status_zh.md` | AD2 提交包状态；当前只作可选附录记录。 |
| `paper_evidence_summary.md` | 历史 diffusion-first 证据快照。 |
| `paper_protocol.md` | 当前论文方法、阈值、数据和声明协议。 |
| `results_limitations_draft.md` | 可直接用于论文的结果与局限性草稿。 |
| `submission_reproducibility_checklist.md` | 投稿与复现检查清单。 |
| `repository_inventory_zh.md` | 本文件，仓库结构与文件作用清单。 |

`paper/`：

| 文件 | 作用 |
|---|---|
| `manuscript.md` | 当前英文论文主稿。 |
| `references.bib` | BibTeX 引用库。 |
| `figures/fig_fixed_vs_oracle_dice.png` | 固定阈值与 oracle Dice 差距图。 |
| `figures/fig_frozen_failure_cases.png` | 冻结协议下的代表性失败案例。 |
| `figures/fig_threshold_fpr_and_oracle_gap.png` | 正常像素 FPR 和阈值乐观偏差图。 |

## 9. 官方 MVTec AD 2 工具

| 文件 | 作用 |
|---|---|
| `official_mvtec_ad2_utils/MVTecAD2_public_code_utils.tar.gz` | 官方工具原始压缩包。 |
| `MVTecAD2_public_code_utils/__init__.py` | 官方 Python 包标记。 |
| `check_and_prepare_data_for_upload.py` | 官方提交完整性检查和打包脚本。 |
| `measure_runtime_and_memory.py` | 官方推理时间和显存测量脚本。 |
| `mvtec_ad_2_public_offline.py` | 官方 public 数据加载与提交写出 API。 |
| `utils.py` | 官方图像、目录、压缩和异常检查函数。 |
| `README.txt` | 官方使用说明。 |
| `requirements.txt` | 官方工具依赖。 |
| `license.txt` | 官方工具许可证。 |

该目录属于外部官方代码，不应按本项目代码风格重写。

## 10. 运行产物 `runs/`

`runs/` 有 2059 个顶层运行目录。前缀表达实验族：

- `feature_*`：当前 feature-first 主实验、候选、融合和固定阈值运行。
- `fewshot_*`：8/16/32-shot、三 seed 实验。
- `official_*`：外部官方/维护实现转换后的统一运行。
- `mvtec15_*`、`visa_*`、`mpdd_*`：早期本地 baseline 和数据集扩展。
- `mvtec_ad2_*`：AD2 public 与模型运行。
- `next_*`、`mini_*`、`recon_*`、`protofix_*`：早期 diffusion-first 门控。
- `retrieval_*`、`audit_*`、`smoke_*`：专项消融、审计和流程验证。

单个运行中常见文件：

| 文件/目录 | 作用 |
|---|---|
| `config.yaml`、`run_args.json` | 完整配置和命令参数。 |
| `git_hash.txt`、`environment.txt` | 代码版本和运行环境。 |
| `feature_prior.pt`、`diffusion.pt`、`hn_sev.pt`、`lc_rds.pt` | 模型权重。 |
| `metrics.json/csv`、`eval_metrics.json` | 汇总指标。 |
| `scores.csv` | 每张图的分数、标签和路径。 |
| `predictions.npz` | 论文重算所需的分数、热图、标签和掩码。 |
| `pixel_threshold_policy.json` | 冻结像素阈值及其选择来源。 |
| `normal_pixel_stats.npz` | 正常图像素校准统计。 |
| `synthetic_validation_seed*.npz/json` | label-free 候选选择证据。 |
| `efficiency.csv`、`pareto.csv` | 延迟、显存、NFE 和 Pareto 结果。 |
| `roi_budget.json/jsonl` | 每个 ROI 的动作、预算和实测延迟。 |
| `heatmaps/`、`masks/`、`repairs/` | 每张样本的输出图。 |
| `images/<case>/` | 输入、重建、残差、最终热图、mask、repair 和 GT 案例。 |

保留优先级：论文主结果对应的 `predictions.npz`、阈值、指标、配置和 provenance
应保留。2026-06-15 已清理 smoke/mini、feature 候选及 retrieval/multiscale
负分支的可视化副本；数值证据和论文定性案例均保留。

## 11. 表格与证据包 `tables/`

`tables/` 有 148 个目录，属于脚本生成结果而非手工源码：

- `feature_first_fusion_aggregate_paper_package/`：当前最权威的综合论文包。
- `feature_first_plan_acceptance/`：19 项验收表与 `summary.json`。
- `strict_fixed_threshold_paper/`：固定阈值主结果和图。
- `external_baseline_comparison/`：七外部 baseline 的主比较。
- `failure_case_panel_mvtec15/`：失败案例选样和面板。
- `diffusionad_compute_plan/`：DiffusionAD 训练时间/存储估算。
- `feature_*`、`heldout_*`、`synthetic_*`、`normal_*`：候选生成、选择、
  稳定性和历史诊断。
- `official_*`：外部方法运行报告和环境审计。
- 根部 `table_main_mvtec.csv`：最早期 MVTec 主表。
- 根部 `table_ablation_hn_sev.csv`：最早期 HN-SEV 消融表。
- 根部 `table_ablation_crv.csv`：最早期 CRV 消融表。
- 根部 `table_ablation_lc_rds.csv`：最早期 LC-RDS 消融表。
- 根部 `table_efficiency.csv`：最早期效率表。这五张表主要用于追溯。

常见文件：

- `summary.json`：该证据包的机器可读结论。
- `table_*.csv`：主表、逐类表、消融表、CI、稳定性或效率表。
- `fig_*.png`：由表格/NPZ 自动生成的论文图。
- `paper_claim_boundary.md`：明确可以和不可以写入论文的声明。
- `gate_summary.json`：是否允许进入下一实验阶段。

## 12. 数据、第三方代码和提交包

`SEER-AD-dataset/`：

| 目录 | 文件数 | 大小 | 作用 |
|---|---:|---:|---|
| `MVTec-AD/` | 8,814 | 4.92 GiB | 主数据集 15 类。 |
| `VisA/` | 12,037 | 1.79 GiB | 主数据集 12 类。 |
| `MPDD/` | 2,484 | 4.42 GiB | 主数据集 6 类，含 1.65 GiB 原始 ZIP。 |
| `MVTec-AD2/` | 8,711 | 30.49 GiB | 可选附录数据集。 |
| `DTD/` | 5,812 | 0.60 GiB | 合成纹理异常来源。 |

DTD 内嵌仓库的 `.git/lfs/incomplete/` 中有 `5623` 个零字节 `.part` 文件，
是一次 Git LFS 下载留下的空占位，不是训练数据。

`third_party/official_baselines/`：

- `patchcore/`、`padim/`、`uniad/`、`draem/`、`ddad/`、`rd4ad/`、
  `simplenet/`、`diffusionad/`：固定 commit 的外部实现。
- DRAEM 约 `10.50 GiB`，主要是作者 checkpoint。
- DDAD 约 `8.96 GiB`，主要是每类预训练特征/模型。
- 其余源码和资产约 `0.36 GiB`。
- 这些文件用于可追溯外部比较，不应混入本项目模块或批量格式化。

`submissions/`：

| 项目 | 大小 | 作用 |
|---|---:|---|
| `mvtec_ad2_seed7/` | 30.51 GiB | 原始分辨率展开提交，4090 TIFF + 4090 PNG。 |
| `mvtec_ad2_seed7.tar.gz` | 8.40 GiB | 上述提交的完整压缩包。 |
| `mvtec_ad2_seed7_model256/` | 0.50 GiB | 256px 紧凑展开提交。 |
| `mvtec_ad2_seed7_model256.tar.gz` | 0.33 GiB | 推荐的 checker-passed 紧凑压缩包。 |
| `*_metadata/` | 很小 | 提交 manifest、协议和哈希。 |

由于 AD2 不再阻塞投稿，原始展开目录和 8.40 GiB 压缩包属于高价值空间清理
候选；紧凑包和 metadata 足以保留未来上传能力。

## 13. 日志和缓存

- `logs/feature_mvtec15_toothbrush_transistor.out.log`：toothbrush/transistor
  批量运行的标准输出。
- `logs/feature_mvtec15_toothbrush_transistor.err.log`：同一批运行的标准错误
  和 warning。
- `logs/feature_mvtec15_wood_zipper.out.log`：wood/zipper 批量运行的标准输出。
- `logs/feature_mvtec15_wood_zipper.err.log`：同一批运行的标准错误和 warning。
- 各目录 `__pycache__/`：Python 缓存。
- `.pytest_cache/`：pytest 缓存。
- 第三方仓库中的 `.git/`：固定源码版本所需；DTD 的内嵌 `.git/` 对训练无用，
  但删除前应确认不再需要更新该数据仓库。

## 14. 当前整理结论

必须保留：

- 根目录脚本、`seer_ad_v2/`、`configs/`、`tools/`、`tests/`、`docs/`、`paper/`。
- 三个主数据集及 DTD 实际图片。
- 当前论文包、19 项验收、七外部 baseline 的指标/预测/provenance。
- 用于论文重算的主运行 NPZ、配置、阈值和指标。

已完成的低风险清理：

- 从 1026 个 smoke/mini、feature 候选和 retrieval/multiscale 负分支运行中
  删除 `488,087` 张 PNG 等可视化图片，释放约 `6.15 GiB`。
- 同时移除 `1,898` 个由此产生的空目录。
- 保留这些运行中的 `38,438` 个 NPZ、`38,174` 个 JSONL，以及所有指标、
  配置、阈值、provenance、权重和日志。
- 保留最终论文修复过程图引用的 4 个 MPDD 运行和 3,680 张源图片。

仍可直接重建、低风险清理：

- 所有 `__pycache__/`、`.pytest_cache/`。
- DTD 中 5623 个零字节 Git LFS `.part`。

需要确认后再清理：

- RD4AD 的 15 个训练 checkpoint，约 16 GiB；删除后指标仍在，但无法直接续训。
- AD2 原始展开提交及 8.40 GiB 压缩包，合计约 38.9 GiB。
- `third_party/` 的 DRAEM/DDAD 权重，删除后重新下载成本较高。
- 历史 `tables/` 和 `runs/`：其中部分是负消融和论文可追溯证据，不能只按旧日期删除。
