# Feature-First 执行记录

## 自动门控

- 数据集：MVTec15、VisA、MPDD。
- 类别数：15 + 12 + 6 = 33。
- seeds：`7,13,23`。
- 选择证据：正常图、合成异常掩码、增强稳定性、正常像素 FPR、实测候选时延。
- 禁止证据：真实异常标签、真实异常掩码。
- 稳定化：对 seeds `7/13/23` 的 synthetic metrics 取均值，冻结同一候选，再分别评估三个 held-out split。
- 新候选：normal-calibrated highres/pixelraw `70/30` fusion；PaDiM 可用类别增加 PaDiM/highres `50/50` fusion。
- 运行状态：9 个 dataset-seed 包全部完成，无同路径对齐缺失。
- 类别选择一致率：`100%`。

## 统计

- category-level paired bootstrap：10,000 次。
- sign test：对 category mean delta 做双侧精确检验。
- 主比较：自动门控结果对每类最强本地 baseline。
- 辅助比较：自动门控结果对同路径 `pixelraw`。

## Few-Shot

- 协议：先冻结 full-data 三 seed 的 dominant policy，再运行 few-shot。
- 训练正常图：`8/16/32`。
- seeds：`7/13/23`。
- 测试：每类固定最多 64 张，不随训练 seed 改变。
- 运行数：15 x 3 x 3 = 135。
- retrieval bank：关闭，原因是正式负消融未通过。

## MVTec AD 2 可选附录资产

- loader 支持：`train`、`validation`、`test_public`、`test_private`、`test_private_mixed`。
- public runner 使用 `train + validation` 正常图训练，不使用 public test 标签配置模型。
- private exporter 输出：
  - 单通道 float16 TIFF 连续异常图；
  - 单通道 0/255 PNG 二值图；
  - 阈值来自 validation normal `mean + 3 * std`。
- 官方 checker 已下载并接入；提交根目录不写入 manifest/protocol，
  元数据保存在相邻目录，避免被判定为多余文件。
- 完整合成提交结构含 `4090` 个私有样本与 `8180` 个 TIFF/PNG 输出，
  已通过官方 checker；checker SHA-256 为
  `c02518512219bf4844a119c6f4f7776716149c3cd932f8bc38cd8fb9d4cbc0f9`。
- public loader 已按官方 `test_public/ground_truth/bad/*_mask.png`
  结构修正，readiness 会强制检查该目录。
- 本机发现器同时检查目录和 ZIP/TAR 内的 8 类完整结构，并可跳过损坏归档。
- 官方 Act-On 表单检查器已接入；提交必须显式提供姓名、邮箱及非商业用途确认，
  Newsletter 默认关闭，状态报告不保存明文邮箱或一次性下载链接。
- ZIP/TAR 安装器只提取官方 8 类目录，拒绝路径穿越、符号链接及向非空目录合并，
  安装后重新检查全部 56 个必需路径。
- 端到端编排器默认只读/dry-run，只有 `--execute` 才运行；private 阶段必须在
  24 个 public 运行完整后开始，并核验 24 个 checkpoint、预测、指标和
  8 类 × 3 seeds 表格覆盖。
- 官方结果导入器拒绝非 `benchmark.mvtec.com` 来源、越界指标、缺失
  submission ID/时间、未通过 checker 的本地提交或不完整 public 24-run 表；
  导入后仍默认 `sota_claim_supported=false`，并生成四数据集统一
  `table_main_seed_metrics_with_ad2.csv`。
- leaderboard 审计器要求我们的 submission 行与官方结果逐项一致，且
  Image AUROC、Pixel AUROC、AUPRO 均严格优于快照内所有对手；并列或任一项
  非第一都不会升级 SOTA 声明。
- 当前状态：数据 56/56 路径完整；public 24/24 运行完成；private 与
  private-mixed 共 4090 张图完成导出；官方 checker 通过；归档为
  推荐上传包为
  `submissions/mvtec_ad2_seed7_model256.tar.gz`（341.67 MiB，官方 checker
  通过）。由于官方网页无法登录，AD2 已从正文主实验和投稿门槛中移除；
  账号上传与服务器结果仅作为未来可选动作。

## 验证

- 单元测试：`124 passed`。
- 计划审计：`62/62 passed`。
- few-shot 运行完整性：`135/135`，缺失为 `0`。
- 计划验收矩阵：`19 pass / 0 fail / 0 blocked / 0 pending_run`，见
  `tables/feature_first_plan_acceptance/`。
- 最终冻结包记录 Python/pip/GPU、git tracked diff、未跟踪文件哈希、
  四份配置、候选池、数据划分和关键论文证据，见
  `tables/protocol_freeze_20260613_final/`。
- 99 个论文选中 `predictions.npz` 均显式保存
  `detection_heatmaps`、`verification_heatmaps`、`image_score_heatmaps`，
  同时保留旧键兼容已有工具。
- 最强对齐本地 baseline 的阈值无关类别胜出：Pixel AP `31/33`，
  AUPRO `30/33`。历史 Dice `27/33` 使用测试 GT oracle 阈值，只保留为
  诊断，不进入论文主声明。
- 99 个 held-out 运行已使用
  `synthetic_normal_fixed_threshold_v1` 重评；最大正常像素 FPR
  `0.4968%`，固定 Dice 相比 oracle Dice 平均低 `0.0706`。
- 33 类独立修复模块审计：2,066 张图像、5,091 个 ROI；异常图
  SSIM `0.9229`、背景 MAE `0.00827`。
- pooled SDR-GT Spearman 为 `-0.1235`，因此 CRV 已正式降级为
  可视化与事后检查模块，不再保留 GT 对齐或验证准确性声明。
- HN-SEV/CRV/LC-RDS 已统一汇总到 33 类：HN-SEV 在 33/33 类降低
  FPRR；LC-RDS 相对 fixed25/rule 均在 33/33 类降低时延，但相对
  fixed10 仅 16/33 类更快。

## 论文材料

- Results/Limitations 草稿：`docs/results_limitations_draft.md`。
- 自动声明边界：`tables/feature_first_fusion_aggregate_paper_package/paper_claim_boundary.md`。
- 贡献链：`fig_contribution_chain.png` 与 `table_contribution_chain.csv`。
- 修复过程图：`fig_repair_process_panel.png`。
- 修复质量与 SDR-GT：`repair_quality_summary.json`、`fig_sdr_gt_correlation.png`。
- 跨数据集模块消融：`module_evidence_summary.json`、
  `table_module_ablation_cross_dataset.csv`。
- 当前所有论文材料均保持“本地对齐 baseline”口径，不提前声明通用外部 SOTA。
