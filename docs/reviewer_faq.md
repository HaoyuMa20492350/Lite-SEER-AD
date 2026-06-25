# Lite-SEER-AD Reviewer FAQ

本文档用于投稿前的审稿质疑预答复。它不新增论文主张，只把仓库中已经冻结的证据边界转成可直接放入 rebuttal、cover letter 或 supplement 的回答。

## Q1. Lite-SEER-AD 是否宣称全面 SOTA？

不宣称。本文主张是 feature-first、label-free、可复核的异常定位闭环：正常图和合成异常用于策略/阈值选择，真实异常标签和掩码只用于 held-out audit。外部基线表保留 PatchCore、PaDiM、UniAD、DRAEM、DDAD、RD4AD、SimpleNet 的对比，但正文只陈述受统计支持的局部优势和限制，不写 universal SOTA。

## Q2. 为什么主线不是 diffusion-first detector？

当前证据支持的主检测器是轻量 feature prior 与 label-free pixel policy。扩散模块只作为选择性局部修复执行器或可视化/audit 组件；它不是主异常检测器，也不是阈值选择依据。这样能避免把计算昂贵且证据不稳定的生成式修复包装成主要检测贡献。

## Q3. CRV 为什么被降级？

CRV 的 repair quality 和 background preservation 有正面证据，但 SDR-GT alignment 没有稳定成立。因此 CRV 不作为反事实验证主贡献，不声明带来 AP/Dice 增益，只保留为 repair visualization、post-hoc audit 和 negative finding。

## Q4. label-free policy 是否使用了真实异常标签或掩码？

没有。选择阶段记录 `uses_real_anomaly_labels=false` 和 `uses_real_anomaly_masks=false`。候选选择依赖正常图、合成异常、normal FPR、synthetic AP/Dice/AUPRO、latency 和 augmentation stability；真实 held-out 异常只在最终 audit 中使用。

## Q5. 图像级 AUROC 为什么不是主声明？

合成域选择的 image-score aggregator 在 held-out audit 上不稳定。仓库已冻结保留当前 `top5` 规则，并把 11 种聚合搜索作为 negative audit。论文主提升应聚焦 AUPRO、Pixel AP、Fixed Dice 等像素定位指标。

## Q6. HN-SEV 还缺什么证据？

已有证据显示 FPRR 在 33/33 类下降。新增 ROI-mask audit 覆盖 33 类、5,091 个 resolved ROI，显示 background/normal ROI suppression 为 `93.01%`，但 GT-overlapping ROI retention 只有 `15.92%`，image-level ROI recall 从 `87.49%` 降到 `15.12%`。因此 HN-SEV 目前只能谨慎表述为 false-positive suppression，不能表述为 recall-safe verifier。

`tables/hn_sev_input_ablation/` 已完成覆盖审计：exact synthetic-only、+clean normal、+hard negative、+feature/prototype 输入消融现在覆盖 33/33 类，并且都有 metric rows。这个结果补齐了证据链，但没有改变 recall-safety 结论；HN-SEV 召回损伤必须作为正式 limitation 保留。

## Q7. LC-RDS 能否声明总是比 fixed10 快？

不能。现有证据支持 LC-RDS 相比 fixed25 和 rule-based repair 更快，但相对 fixed10 只有很弱的平均延迟优势且不是每类都快。最终表述应绑定预算扫描、budget violation rate 和 Pareto area，不写普遍快于 fixed10。

`tables/lc_rds_budget_audit/table_action_space.csv` 进一步区分 required/config/code/observed 动作。当前 v2 配置和 scheduler code 已支持 `skip`、`repair-5`、`repair-10`、`repair-25`、`native-refine`。`tables/lc_rds_budget_sweep/` 保留 measured synthetic action sweep 和 33 类 ROI measured-latency replay；`tables/deployment_production_latency/` 现在给出真实 frozen production 六预算 sweep：`budget_runs=198/198`、`missing_budget_runs=0`、五个 required actions 全部观察到，最大 budget violation rate 为 `0.0009206`。

因此 LC-RDS 可以写成已实现五动作预算空间、offline replay audit、synthetic action execution audit、measured-latency ROI replay，以及真实 production 六预算 multi-action sweep；但仍不能写成普遍快于 fixed10，也不能把部署速度声明外推到未测硬件。

## Q8. 为什么 DiffusionAD full 不在主表？

官方 15 类 3000 epoch 复现成本极高，当前估算约数千 GPU 小时。默认投稿策略不使用 DiffusionAD-Lite 或 smoke 结果冒充 official；DiffusionAD full 仅作为 P2 条件项，在目标期刊或审稿策略强要求时执行。

## Q9. 为什么不用 AE/inpainting/nearest patch 替代扩散修复？

现在已经可以排除“扩散必需”这个强主张。clean-target synthetic-defect audit 覆盖 33 类正常图，已包含 simple inpainting、trained linear partial-conv、partial-conv proxy、nearest normal patch、trained PCA light AE、light AE proxy、light U-Net proxy，以及同协议 category diffusion executor，并已启用真实 LPIPS。当前质量最强的非扩散执行器是 `partial_conv_inpaint`：mean PSNR `37.12`、mean SSIM `0.9826`、LPIPS `0.0191`、foreground MAE `0.0696`；同协议 diffusion 的 mean PSNR 为 `21.73`、mean SSIM 为 `0.9378`、LPIPS 为 `0.1319`、mean latency 为 `67.94 ms`，弱于 `partial_conv_inpaint` 的 `12.32 ms`。因此 Pareto decision 是 `diffusion_not_necessary`。

审稿时应主动说明：扩散模块可保留为可替换 repair executor 或可视化组件，但不再作为核心必要性卖点。主贡献应落在 label-free feature-first detector、HN-SEV false-positive suppression 和 LC-RDS budget scheduling。

## Q10. 如何证明结果可复现？

仓库包含 `requirements-lock.txt`、`environment.yml`、`Dockerfile`、CI、`CITATION.cff`、`.zenodo.json`、prediction manifest、fixed-threshold bundle 和 SHA256 artifact manifest。投稿前需要重新生成 manifest，并在 GitHub Release、Zenodo DOI、Hugging Face model/dataset card 发布后回填链接。

`tables/release_readiness/summary.json` 现在把本地工件和外部发布分开审计：本地 `local_artifact_ready=true`，但 `release_gate_passed=false`，因为 `github_release_url`、`zenodo_doi`、`hf_model_url`、`hf_dataset_url` 还没有真实值。发布后需要填写 `release_metadata.json`，运行 `python tools/render_release_metadata.py --input release_metadata.json` 生成 `release_links.json`、`CITATION.cff` 和 `.zenodo.json`，再重新运行 `python tools/export_release_readiness.py`。

## Q11. 能否声明已经完成部署级延迟评估？

还不能。`tables/deployment_latency/` 已经提供 synchronized batch=1 component smoke audit，包含 IO、detector、verifier、scheduler、repair、end-to-end proxy 的 mean/p50/p95/p99 和硬件信息；当前 RTX 4090 Laptop GPU 环境下 smoke end-to-end proxy 的 mean latency 是 `0.3600 ms`、p95 是 `0.4480 ms`、p99 是 `0.4737 ms`。

`tables/lc_rds_budget_sweep/` 还补充了 measured synthetic action sweep：`skip`、`repair-5`、`repair-10`、`repair-25`、`native-refine` 均有 p95/p99 动作延迟，synthetic budget violation rate 为 `0.0`；并将实测 action p95 延迟回放到 33 类 ROI 日志，ROI measured budget violation rate 为 `0.0`。`tables/deployment_production_latency/` 进一步汇总了 33 个真实 `utility_lc_rds` source inference runs 和 198 个 guarded production budget runs；当前 summary 记录 `231` 个 runs、`30415` 张图的 detector/HN-SEV verifier/repair/end-to-end component latency。

但这仍不是最终 production deployment claim。真实六预算 sweep 已完成：33 类乘 6 个预算点全部覆盖，`missing_budget_runs=0`，五个 required actions 全部观察到，最大 budget violation rate 为 `0.0009206`，低于 1% gate。`tables/deployment_production_latency/energy_measurements/` 还记录了当前硬件上的 production-style 能耗 probe：8 张 `mvtec15/bottle` budget-10 图像约 `274.41 J`。剩余部署 blocker 是跨硬件验证。因此速度声明可以写成“当前硬件上的 production budget sweep ready”，但不能写成跨硬件部署强结论。

## Q12. 论文失败案例如何处理？

弱类别和失败模块不隐藏。`tables/failure_taxonomy/` 已覆盖 grid、screw、pill/capsule、hazelnut、transistor、fryum、bracket_white 等保留弱类，并明确标注为 post-hoc failure analysis only；真实 mask 只用于诊断归因，不用于方法、候选或阈值选择。若未来要把这些弱类从 limitation 推进为性能增强，需要新增 label-free 专项候选并重新冻结。
