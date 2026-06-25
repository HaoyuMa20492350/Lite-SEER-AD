"""Export the Lite-SEER-AD 100% completion gap matrix.

The matrix is intentionally evidence driven: it records what the current
repository can prove, what is only solved by a declared limitation or negative
finding, and what still needs new experiments or journal-specific metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path("tables/completion_gap_matrix")
DEFAULT_DATE = "2026-06-21"

BLOCKING_PRIORITIES = {"P0"}
NON_BLOCKING_STATUSES = {
    "complete",
    "complete_with_declared_limit",
    "optional_not_required",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def has_file(root: Path, relative: str) -> bool:
    return (root / relative).is_file()


def has_any(root: Path, patterns: list[str]) -> bool:
    return any(any(root.glob(pattern)) for pattern in patterns)


def evidence(*paths: str) -> str:
    return "; ".join(paths)


def make_row(
    *,
    priority: str,
    dimension: str,
    target_100: str,
    status: str,
    current_completion_percent: int,
    current_evidence: str,
    missing_to_100: str,
    implementation_action: str,
    acceptance_criteria: str,
) -> dict[str, Any]:
    return {
        "priority": priority,
        "dimension": dimension,
        "target_100": target_100,
        "status": status,
        "current_completion_percent": current_completion_percent,
        "blocking_for_default_100": (
            priority in BLOCKING_PRIORITIES and status not in NON_BLOCKING_STATUSES
        ),
        "current_evidence": current_evidence,
        "missing_to_100": missing_to_100,
        "implementation_action": implementation_action,
        "acceptance_criteria": acceptance_criteria,
    }


def plan_acceptance_complete(plan: dict[str, Any]) -> bool:
    counts = plan.get("counts", {}) or {}
    return (
        plan.get("complete") is True
        and counts.get("pass", 0) >= 19
        and counts.get("fail", 1) == 0
        and counts.get("blocked", 1) == 0
        and counts.get("pending_run", 1) == 0
    )


def external_baselines_complete(summary: dict[str, Any]) -> bool:
    methods = set(summary.get("methods", []) or [])
    required = {
        "patchcore",
        "padim",
        "uniad",
        "draem",
        "ddad",
        "rd4ad",
        "simplenet",
    }
    return (
        required.issubset(methods)
        and summary.get("uses_real_anomaly_labels_for_threshold") is False
        and summary.get("uses_real_anomaly_masks_for_threshold") is False
    )


def prediction_release_ready(predictions: dict[str, Any], root: Path) -> bool:
    return (
        predictions.get("all_predictions_present") is True
        and predictions.get("all_threshold_policies_present") is True
        and has_file(root, "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json")
        and has_file(root, "artifacts/manifest.json")
    )


def build_rows(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()

    plan = read_json(root / "tables/feature_first_plan_acceptance/summary.json")
    fusion = read_json(
        root / "tables/feature_first_fusion_aggregate_paper_package/summary.json"
    )
    modules = read_json(
        root
        / "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json"
    )
    repair = read_json(
        root
        / "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json"
    )
    image_agg = read_json(root / "tables/image_score_aggregation_mvtec15/summary.json")
    external = read_json(root / "tables/external_baseline_comparison/summary.json")
    predictions = read_json(root / "artifacts/predictions_manifest.json")

    manuscript = read_text(root / "paper/manuscript.md")
    claim_freeze = read_text(root / "docs/claim_freeze_2026_06_21.md")
    has_reviewer_faq = has_file(root, "docs/reviewer_faq.md")
    has_statement_placeholders = has_file(root, "docs/submission_statement_placeholders.md")

    rows: list[dict[str, Any]] = []

    manuscript_lower = manuscript.lower()
    manuscript_normalized = " ".join(manuscript_lower.split())
    forbidden_universal_sota = "universal sota" in manuscript_normalized and not any(
        phrase in manuscript_normalized
        for phrase in [
            "do not claim universal sota",
            "does not claim universal sota",
            "not claim universal sota",
            "no universal sota",
        ]
    )
    scientific_frozen = (
        "Feature-first Lite-SEER-AD" in claim_freeze
        and "Label-Free Feature-First" in manuscript
        and not forbidden_universal_sota
        and (
            "crv is used only" in manuscript_normalized
            or "crv is restricted to visualization" in manuscript_normalized
        )
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="科学定位",
            target_100=(
                "题目、摘要、贡献和方法图统一为 feature-first + label-free + "
                "selective repair，不把扩散或 CRV 写成主检测贡献。"
            ),
            status="complete" if scientific_frozen else "partial",
            current_completion_percent=100 if scientific_frozen else 70,
            current_evidence=evidence(
                "docs/claim_freeze_2026_06_21.md",
                "paper/manuscript.md",
                "tables/feature_first_fusion_aggregate_paper_package/paper_claim_boundary.md",
            ),
            missing_to_100=(
                "无默认 P0 缺口；投稿前只需按目标期刊版式同步标题、摘要和贡献表述。"
                if scientific_frozen
                else "需要移除 diffusion-first、CRV 主贡献或 universal SOTA 叙事。"
            ),
            implementation_action=(
                "保持三条贡献冻结：label-free policy gate、HN-SEV、LC-RDS；"
                "CRV 只放在 visualization/audit/limitation。"
            ),
            acceptance_criteria=(
                "全文不宣称扩散主检测，不宣称 universal SOTA，不把 CRV 写入主贡献。"
            ),
        )
    )

    label_free_complete = (
        plan_acceptance_complete(plan)
        and fusion.get("category_selection_agreement", 0) >= 0.95
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="主检测器与 label-free 像素策略",
            target_100=(
                "synthetic-normal selection 作为论文级 policy gate，选择过程只用正常图、"
                "合成异常、normal FPR、synthetic AP/Dice/AUPRO、latency 和 stability。"
            ),
            status="complete" if label_free_complete else "partial",
            current_completion_percent=100 if label_free_complete else 70,
            current_evidence=evidence(
                "tables/feature_first_plan_acceptance/summary.json",
                "tables/synthetic_gate_fusion_aggregate_*/seed*/selection.csv",
                "tables/feature_first_fusion_aggregate_paper_package/table_paired_bootstrap_ci.csv",
                "seer_ad_v2/evaluation/pixel_threshold_policy.py",
            ),
            missing_to_100=(
                "无默认 P0 缺口；继续保留 uses_real_anomaly_labels=false 和 "
                "uses_real_anomaly_masks=false 的记录。"
                if label_free_complete
                else "需要补齐 99 个 selected candidate 的无真异常标签/掩码选择记录和正向 CI。"
            ),
            implementation_action=(
                "冻结候选池与 fixed threshold JSON；所有新候选必须通过相同 gate 重新导出。"
            ),
            acceptance_criteria=(
                "每个 selected candidate 记录 label/mask flags=false；AUPRO/Pixel AP 的 "
                "BCa 95% CI 相对 feature_raw 不跨 0。"
            ),
        )
    )

    image_agg_complete = (
        image_agg.get("adoption_decision") == "retain_current_top5"
        and image_agg.get("uses_real_anomaly_labels_for_selection") is False
        and image_agg.get("uses_real_anomaly_masks_for_selection") is False
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="图像级聚合",
            target_100=(
                "保留冻结 top5 或预声明规则；11 种聚合搜索只作为负结果/审计，"
                "不能按 held-out 结果事后切换。"
            ),
            status="complete_with_declared_limit" if image_agg_complete else "partial",
            current_completion_percent=100 if image_agg_complete else 65,
            current_evidence=evidence(
                "tables/image_score_aggregation_mvtec15/summary.json",
                "tables/image_score_aggregation_mvtec15/analysis.md",
            ),
            missing_to_100=(
                "无默认 P0 缺口；Image AUROC 不作为主提升声明。"
                if image_agg_complete
                else "需要冻结 aggregator 或明确 negative audit，不得继续调参切换阈值。"
            ),
            implementation_action=(
                "正文把聚合搜索放入 audit/negative result；主表聚焦像素定位指标。"
            ),
            acceptance_criteria=(
                "若 label-free 聚合选择低于 top5，按 negative audit 报告，不作为主提升。"
            ),
        )
    )

    hn = modules.get("hn_sev", {}) or {}
    hn_fpr_ready = (
        hn.get("positive_repeatable_evidence") is True
        and hn.get("categories_with_lower_fprr", 0) >= 33
    )
    hn_audit = read_json(root / "tables/hn_sev_retention_calibration/summary.json")
    hn_input = read_json(root / "tables/hn_sev_input_ablation/summary.json")
    hn_input_audit_ready = (
        hn_input.get("evidence_level")
        in {
            "historical_ablation_table_coverage_v1",
            "mixed_table_run_metadata_coverage_v2",
        }
        and hn_input.get("categories", 0) > 0
    )
    hn_input_full_ready = hn_input.get("release_gate_passed") is True
    hn_overall = hn_audit.get("overall", {}) or {}
    hn_retention_ready = (
        hn_audit.get("evidence_level") == "roi_mask_retention_calibration_v1"
        and hn_audit.get("categories", 0) >= 33
        and hn_overall.get("roi_rows_resolved", 0) > 0
    )
    hn_tp_retention = float(hn_overall.get("tp_retention", 0.0) or 0.0)
    hn_recall_before = float(hn_overall.get("roi_recall_before_hn_sev", 0.0) or 0.0)
    hn_recall_after = float(hn_overall.get("roi_recall_after_hn_sev", 0.0) or 0.0)
    hn_background_suppression = float(
        hn_overall.get("background_suppression_rate", 0.0) or 0.0
    )
    hn_retention_recall_safe = (
        hn_retention_ready
        and hn_tp_retention >= 0.95
        and hn_recall_after >= hn_recall_before - 0.02
    )
    hn_recall_limitation = (
        hn_retention_ready
        and not hn_retention_recall_safe
        and "recall-safety limitation" in read_text(
            root / "docs/results_limitations_draft.md"
        ).lower()
    )
    hn_full_ready = hn_audit.get("release_gate_passed") is True and hn_input_full_ready
    hn_claim_bounded_ready = (
        hn_fpr_ready
        and hn_retention_ready
        and hn_input_full_ready
        and (hn_retention_recall_safe or hn_recall_limitation)
    )
    hn_partial_pct = (
        88
        if hn_retention_ready and hn_input_audit_ready
        else (85 if hn_retention_ready else (65 if hn_fpr_ready else 30))
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="HN-SEV",
            target_100=(
                "证明 FPRR 降低不是靠删除 ROI：补 TP retention、ROI recall、ECE/Brier、"
                "hard-negative taxonomy 和输入消融。"
            ),
            status=(
                "complete"
                if hn_full_ready
                else (
                    "complete_with_declared_limit"
                    if hn_claim_bounded_ready
                    else ("partial" if hn_fpr_ready else "pending")
                )
            ),
            current_completion_percent=(
                100
                if hn_full_ready or hn_claim_bounded_ready
                else hn_partial_pct
            ),
            current_evidence=evidence(
                "tables/hn_sev_retention_calibration/summary.json",
                "tables/hn_sev_retention_calibration/table_category_hn_sev_audit.csv",
                "tables/hn_sev_retention_calibration/table_before_after_cases.csv",
                "tables/hn_sev_input_ablation/summary.json",
                "tables/hn_sev_input_ablation/table_input_ablation_coverage.csv",
                "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
                "seer_ad_v2/models/region_verifier/hn_sev.py",
            ),
            missing_to_100=(
                "无缺口。"
                if hn_full_ready
                else (
                    "无默认 P0 缺口；exact 输入消融、retention/calibration/taxonomy "
                    "均已齐备。HN-SEV 只能声明 false-positive suppression，"
                    f"不能声明 recall-safe verifier：当前 TP retention={hn_tp_retention:.2%}, "
                    f"ROI recall {hn_recall_before:.2%}->{hn_recall_after:.2%}。"
                    if hn_claim_bounded_ready
                    else (
                        "已有 TP retention、ROI recall、ECE/Brier、hard-negative taxonomy "
                        f"和 before/after case index；当前 background suppression={hn_background_suppression:.2%}, "
                        f"TP retention={hn_tp_retention:.2%}, ROI recall {hn_recall_before:.2%}->{hn_recall_after:.2%}，"
                        "已在 limitation 中限制为 false-positive suppression；仍缺 exact synthetic-only/"
                        "+clean normal/+hard negative/+feature/prototype 33 类输入消融。"
                        if hn_recall_limitation and hn_input_audit_ready
                        else
                        "已有 TP retention、ROI recall、ECE/Brier、hard-negative taxonomy "
                        "和 before/after case index；输入消融已审计但 exact synthetic-only/"
                        "+clean normal/+hard negative/+feature/prototype 仍未 33 类齐备。"
                        if hn_retention_ready and hn_input_audit_ready
                        else "已有 TP retention、ROI recall、ECE/Brier、hard-negative taxonomy "
                        "和 before/after case index；仍缺 synthetic-only/+clean normal/"
                        "+hard negative/+feature/prototype 输入消融。"
                        if hn_retention_ready
                        else "缺 TP retention、ROI recall、ECE/Brier、hard-negative 分型和 "
                        "synthetic-only/+clean normal/+hard negative/+feature/prototype 消融。"
                    )
                )
            ),
            implementation_action=(
                "保持 HN-SEV retention/calibration/taxonomy/input-ablation 审计；"
                "若未来要声明 recall-safe，必须重新设计 HN-SEV 并恢复 TP retention/ROI recall。"
            ),
            acceptance_criteria=(
                "FPRR 下降且 TP retention 不显著下降；若召回受损，作为 limitation。"
            ),
        )
    )

    lc = modules.get("lc_rds", {}) or {}
    lc_high_budget_ready = (
        lc.get("positive_repeatable_high_budget_evidence") is True
        and lc.get("fixed25", {}).get("categories_faster", 0) >= 33
        and lc.get("rule", {}).get("categories_faster", 0) >= 33
    )
    budget_summary = read_json(root / "tables/lc_rds_budget_sweep/summary.json")
    budget_audit = read_json(root / "tables/lc_rds_budget_audit/summary.json")
    production_budget_summary = read_json(root / "tables/deployment_production_latency/summary.json")
    measured_budget_full_ready = (
        bool(budget_summary)
        and budget_summary.get("evidence_level") == "measured_budget_sweep_v1"
        and budget_summary.get("release_gate_passed") is True
        and budget_summary.get("budgets_ms") == [10, 25, 50, 75, 100, 150]
        and budget_summary.get("max_budget_violation_rate", 1.0) <= 0.01
    )
    production_budget_full_ready = (
        bool(production_budget_summary)
        and production_budget_summary.get("release_gate_passed") is True
        and production_budget_summary.get("multi_action_budget_sweep_ready") is True
        and production_budget_summary.get("budget_sweep_coverage_ready") is True
        and production_budget_summary.get("budgets_ms") == [10, 25, 50, 75, 100, 150]
        and production_budget_summary.get("missing_budget_runs") == 0
        and production_budget_summary.get("budget_runs")
        == production_budget_summary.get("expected_budget_runs")
        and set(production_budget_summary.get("observed_actions", []) or [])
        >= {"skip", "repair-5", "repair-10", "repair-25", "native-refine"}
        and production_budget_summary.get("max_budget_violation_rate", 1.0) <= 0.01
    )
    budget_full_ready = measured_budget_full_ready or production_budget_full_ready
    budget_synthetic_ready = (
        bool(budget_summary)
        and budget_summary.get("evidence_level")
        == "measured_synthetic_action_budget_sweep_v1"
        and budget_summary.get("budgets_ms") == [10, 25, 50, 75, 100, 150]
        and budget_summary.get("action_coverage_ready") is True
        and budget_summary.get("max_budget_violation_rate", 1.0) <= 0.01
    )
    budget_roi_measured_ready = (
        budget_synthetic_ready
        and budget_summary.get("roi_measured_budget_replay_ready") is True
        and budget_summary.get("roi_measured_categories", 0) >= 33
        and budget_summary.get("max_roi_measured_budget_violation_rate", 1.0) <= 0.01
    )
    budget_replay_ready = (
        bool(budget_audit)
        and budget_audit.get("budgets_ms") == [10, 25, 50, 75, 100, 150]
        and budget_audit.get("evidence_level") == "offline_replay_from_roi_logs_v1"
    )
    budget_action_space = budget_audit.get("action_space", {}) or {}
    budget_action_ready = (
        budget_replay_ready
        and budget_action_space.get("action_space_ready") is True
        and not budget_action_space.get("missing_code_actions")
        and not budget_action_space.get("missing_config_actions")
    )
    if budget_roi_measured_ready:
        lc_partial_pct = 90
    elif budget_synthetic_ready:
        lc_partial_pct = 88
    elif budget_action_ready:
        lc_partial_pct = 84
    elif budget_replay_ready:
        lc_partial_pct = 80
    elif lc_high_budget_ready:
        lc_partial_pct = 65
    else:
        lc_partial_pct = 35
    if budget_full_ready:
        lc_missing_to_100 = (
            "无缺口；真实 frozen production 六预算 sweep 已完成，"
            f"budget_runs={production_budget_summary.get('budget_runs')}, "
            f"missing_budget_runs={production_budget_summary.get('missing_budget_runs')}, "
            f"max_budget_violation_rate={production_budget_summary.get('max_budget_violation_rate')}。"
            if production_budget_full_ready
            else "无缺口。"
        )
    elif budget_roi_measured_ready:
        lc_missing_to_100 = (
            "已有 offline ROI replay、required action-space 审计、measured synthetic "
            "action sweep，以及基于实测 action p95 的 33 类 ROI budget replay；"
            "但历史 ROI logs 仍只观察到 repair-10，仍缺真实 frozen pipeline 的 "
            "multi-action per-budget synchronized sweep。"
        )
    elif budget_synthetic_ready:
        lc_missing_to_100 = (
            "已有 offline ROI replay、required action-space 审计和 measured synthetic "
            "action budget sweep，五个 required actions 均可执行且预算违规率 <=1%；"
            "仍缺真实 frozen pipeline 的 per-budget synchronized sweep。"
        )
    elif budget_action_ready:
        lc_missing_to_100 = (
            "已有 offline ROI replay 和 required action-space 审计；仍缺真实 per-budget "
            "measured synchronized sweep，且历史 ROI logs 未必观察到所有 required actions。"
        )
    elif budget_replay_ready:
        lc_missing_to_100 = (
            "已有 offline ROI replay，但仍缺 measured synchronized budget sweep；"
            "p95/p99、budget violation、gain/ms、Pareto area 不能作为部署强声明。"
        )
    else:
        lc_missing_to_100 = (
            "缺多预算 sweep 的 p95/p99、budget violation、gain/ms、Pareto area；"
            "不能声明普遍快于 fixed10。"
        )
    rows.append(
        make_row(
            priority="P0",
            dimension="LC-RDS",
            target_100=(
                "固定预算 10/25/50/75/100/150 ms，动作覆盖 skip/repair5/repair10/"
                "repair25/native_refine，报告完整延迟、NFE、修复面积、违规率、gain/ms 和 Pareto。"
            ),
            status=(
                "complete"
                if budget_full_ready
                else ("partial" if lc_high_budget_ready else "pending")
            ),
            current_completion_percent=(
                100
                if budget_full_ready
                else lc_partial_pct
            ),
            current_evidence=evidence(
                "configs/scheduler/lc_rds_budget_v2.yaml",
                "tables/lc_rds_budget_audit/summary.json",
                "tables/lc_rds_budget_audit/table_action_space.csv",
                "tables/lc_rds_budget_sweep/summary.json",
                "tables/lc_rds_budget_sweep/table_action_latency_summary.csv",
                "tables/lc_rds_budget_sweep/table_budget_sweep.csv",
                "tables/lc_rds_budget_sweep/table_roi_measured_budget_sweep.csv",
                "tables/deployment_production_latency/summary.json",
                "tables/deployment_production_latency/table_production_budget_replay.csv",
                "tables/deployment_production_latency/table_production_action_coverage.csv",
                "tables/feature_first_fusion_aggregate_paper_package/module_evidence_summary.json",
                "tables/feature_first_fusion_aggregate_paper_package/table_efficiency_pareto.csv",
            ),
            missing_to_100=lc_missing_to_100,
            implementation_action=(
                "运行 LC-RDS production budget sweep 并写出 "
                "tables/deployment_production_latency/summary.json 和 per-budget CSV。"
            ),
            acceptance_criteria=(
                "同预算更好或同质量更快；budget violation rate <= 1%；fixed10 只作限制性描述。"
            ),
        )
    )

    crv_downgraded = (
        repair.get("claim_decision") == "downgrade_to_visualization_only"
        or "Visualization only" in (modules.get("crv", {}) or {}).get("claim", "")
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="CRV",
            target_100=(
                "默认降级为 repair visualization / post-hoc audit；不作为反事实验证主贡献。"
            ),
            status="complete_with_declared_limit" if crv_downgraded else "partial",
            current_completion_percent=100 if crv_downgraded else 55,
            current_evidence=evidence(
                "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json",
                "tables/feature_first_fusion_aggregate_paper_package/table_sdr_gt_correlation_summary.csv",
                "seer_ad_v2/models/counterfactual/repair_verification.py",
            ),
            missing_to_100=(
                "无默认 P0 缺口；若未来抢救 CRV，必须新增跨数据集 Spearman 转正证据。"
                if crv_downgraded
                else "需要从摘要、贡献、主图、结论中移除 CRV 主贡献定位。"
            ),
            implementation_action=(
                "保留 repair quality、background preservation 和 SDR-GT negative finding。"
            ),
            acceptance_criteria=(
                "明确写 CRV 不支持 GT-aligned repair，也不支持 AP/Dice 增益。"
            ),
        )
    )

    repair_executor_ablation = read_json(
        root / "tables/repair_executor_ablation/summary.json"
    )
    repair_pareto = repair_executor_ablation.get("diffusion_pareto_decision", {}) or {}
    diffusion_necessity_ready = (
        repair_executor_ablation.get("release_gate_passed") is True
        and repair_pareto.get("ready") is True
    )
    non_diffusion_ablation_ready = (
        repair_executor_ablation.get("evidence_level")
        == "clean_target_non_diffusion_executor_ablation_v1"
        and repair_executor_ablation.get("images", 0) > 0
    )
    executor_coverage = repair_executor_ablation.get("executor_family_coverage", {}) or {}
    executor_family_proxy_ready = (
        non_diffusion_ablation_ready
        and executor_coverage.get("simple_inpainting_ready") is True
        and executor_coverage.get("partial_conv_proxy_ready") is True
        and executor_coverage.get("nearest_normal_patch_ready") is True
        and executor_coverage.get("light_ae_proxy_ready") is True
        and executor_coverage.get("light_unet_proxy_ready") is True
    )
    repair_lpips_ready = executor_coverage.get("lpips_metric_ready") is True
    trained_light_ae_ready = executor_coverage.get("trained_light_ae_ready") is True
    trained_partial_or_unet_ready = (
        executor_coverage.get("trained_light_unet_or_partial_conv_ready") is True
    )
    repair_executor_pct = (
        82
        if executor_family_proxy_ready
        and repair_lpips_ready
        and trained_light_ae_ready
        and trained_partial_or_unet_ready
        else
        76
        if executor_family_proxy_ready and repair_lpips_ready and trained_light_ae_ready
        else
        72
        if executor_family_proxy_ready and repair_lpips_ready
        else (68 if executor_family_proxy_ready else (60 if non_diffusion_ablation_ready else 30))
    )
    if diffusion_necessity_ready:
        repair_missing_to_100 = "无缺口。"
    elif (
        executor_family_proxy_ready
        and repair_lpips_ready
        and trained_light_ae_ready
        and trained_partial_or_unet_ready
    ):
        repair_missing_to_100 = (
            "已有 clean-target 非扩散执行器对照，覆盖 simple inpainting、trained "
            "partial-conv、nearest normal patch、trained PCA light AE、light U-Net "
            "proxy，并已导出真实 LPIPS；仍缺同协议 diffusion executor 和最终 Pareto 判定。"
        )
    elif executor_family_proxy_ready and repair_lpips_ready and trained_light_ae_ready:
        repair_missing_to_100 = (
            "已有 clean-target 非扩散执行器对照，覆盖 simple inpainting、partial-conv "
            "proxy、nearest normal patch、trained PCA light AE、light U-Net proxy，并已导出"
            "真实 LPIPS；仍缺同协议 diffusion executor、trained light U-Net 或 "
            "trained partial-conv 和最终 Pareto 判定。"
        )
    elif executor_family_proxy_ready and repair_lpips_ready:
        repair_missing_to_100 = (
            "已有 clean-target 非扩散执行器对照，覆盖 simple inpainting、partial-conv "
            "proxy、nearest normal patch、light AE proxy、light U-Net proxy，并已导出"
            "真实 LPIPS；仍缺同协议 diffusion executor、trained light AE/light U-Net "
            "或 partial-conv 和最终 Pareto 判定。"
        )
    elif executor_family_proxy_ready:
        repair_missing_to_100 = (
            "已有 clean-target 非扩散执行器对照，并覆盖 simple inpainting、"
            "partial-conv proxy、nearest normal patch、light AE proxy、"
            "light U-Net proxy；仍缺同协议 diffusion executor、trained light "
            "AE/light U-Net 或 partial-conv、LPIPS 和最终 Pareto 判定。"
        )
    elif non_diffusion_ablation_ready:
        repair_missing_to_100 = (
            "已有 clean-target 非扩散执行器对照；仍缺 partial-conv/light U-Net "
            "覆盖、同协议 diffusion executor、trained light AE/light U-Net 或 "
            "partial-conv、LPIPS 和最终 Pareto 判定。"
        )
    else:
        repair_missing_to_100 = "缺非扩散执行器对照的 PSNR/SSIM/LPIPS/background error/latency Pareto。"
    rows.append(
        make_row(
            priority="P0",
            dimension="扩散修复必要性",
            target_100=(
                "用 light AE、partial/simple inpainting、nearest normal patch、light U-Net "
                "等非扩散执行器回答为什么需要 diffusion repair。"
            ),
            status=(
                "complete"
                if diffusion_necessity_ready
                else ("partial" if non_diffusion_ablation_ready else "pending")
            ),
            current_completion_percent=(
                100
                if diffusion_necessity_ready
                else repair_executor_pct
            ),
            current_evidence=evidence(
                "tables/repair_executor_ablation/summary.json",
                "tables/repair_executor_ablation/table_repair_executor_summary.csv",
                "tables/repair_executor_ablation/table_repair_executor_images.csv",
                "tables/repair_executor_ablation/table_executor_family_coverage.csv",
                "tables/feature_first_fusion_aggregate_paper_package/repair_quality_summary.json",
                "seer_ad_v2/models/counterfactual",
            ),
            missing_to_100=repair_missing_to_100,
            implementation_action=(
                "在 clean-target synthetic defect 协议下跑 repair executor ablation；"
                "若 diffusion 不占 Pareto 优势，降级为可替换执行器。"
            ),
            acceptance_criteria=(
                "diffusion Pareto 更优才保留为 repair executor；否则不得把 diffusion 当核心卖点。"
            ),
        )
    )

    baseline_ready = external_baselines_complete(external)
    rows.append(
        make_row(
            priority="P0",
            dimension="外部基线与统计",
            target_100=(
                "外部 baseline、paired BCa bootstrap、sign test、Holm correction、Spearman "
                "全部可审计；DiffusionAD full 只作 P2 条件项。"
            ),
            status="complete_with_declared_limit" if baseline_ready else "partial",
            current_completion_percent=95 if baseline_ready else 60,
            current_evidence=evidence(
                "tables/external_baseline_comparison/summary.json",
                "artifacts/stats/*.json",
                "baselines/official_sources.json",
            ),
            missing_to_100=(
                "无默认 P0 缺口；DiffusionAD full 不得用 Lite/smoke 代替 official。"
                if baseline_ready
                else "需要补齐 7 个外部 baseline 的来源、commit、paired statistics 和 label-free threshold 记录。"
            ),
            implementation_action=(
                "维持 7 个已完成外部 baseline；DiffusionAD full 只在目标期刊/审稿强要求时执行。"
            ),
            acceptance_criteria=(
                "所有显著性声明有 CI/校正 p；不使用 DiffusionAD-Lite/smoke 冒充 official。"
            ),
        )
    )

    weak_specialization_ready = has_file(
        root,
        "tables/weak_category_specialization/summary.json",
    )
    weak_taxonomy = read_json(root / "tables/failure_taxonomy/summary.json")
    weak_taxonomy_ready = (
        weak_taxonomy.get("release_gate_passed") is True
        and weak_taxonomy.get("all_required_weak_categories_covered") is True
    )
    weak_category_ready = weak_specialization_ready or weak_taxonomy_ready
    rows.append(
        make_row(
            priority="P1",
            dimension="弱类别",
            target_100=(
                "grid、screw、pill/capsule、hazelnut、transistor、fryum、bracket_white "
                "不再是说服力风险，或被纳入 failure taxonomy。"
            ),
            status=(
                "complete"
                if weak_specialization_ready
                else "complete_with_declared_limit"
                if weak_taxonomy_ready
                else "partial"
            ),
            current_completion_percent=100 if weak_category_ready else 55,
            current_evidence=evidence(
                "tables/failure_taxonomy/summary.json",
                "tables/failure_taxonomy/table_failure_taxonomy.csv",
                "tables/failure_taxonomy/analysis.md",
                "tables/failure_case_panel_mvtec15",
                "docs/current_stage_and_next_plan_zh.md",
            ),
            missing_to_100=(
                "弱类已作为 post-hoc failure taxonomy 完整限界；若未来要把 P1 从声明边界推进为性能增强，"
                "仍需新增 label-free 专项候选并重新冻结。"
                if weak_taxonomy_ready and not weak_specialization_ready
                else "无缺口。"
                if weak_specialization_ready
                else "缺弱类专项 gate 输出：grid 频域/空间校准、screw highres ROI、"
                "pill/capsule 颜色/局部对比、bracket_white local contrast normalization。"
            ),
            implementation_action=(
                "把弱类候选纳入同一 label-free gate；失败时输出 failure taxonomy 而非人工挑选。"
            ),
            acceptance_criteria=(
                "最差类 Fixed Dice 不再灾难性为 0；若仍失败，正文/补充材料给 taxonomy。"
            ),
        )
    )

    deployment_summary = read_json(root / "tables/deployment_latency/summary.json")
    deployment_readiness = read_json(root / "tables/deployment_readiness/summary.json")
    deployment_budget_sweep = read_json(root / "tables/lc_rds_budget_sweep/summary.json")
    deployment_production = read_json(root / "tables/deployment_production_latency/summary.json")
    deployment_synthetic_budget_ready = (
        deployment_budget_sweep.get("evidence_level")
        == "measured_synthetic_action_budget_sweep_v1"
        and deployment_budget_sweep.get("action_coverage_ready") is True
        and deployment_budget_sweep.get("max_budget_violation_rate", 1.0) <= 0.01
    )
    deployment_roi_measured_ready = (
        deployment_synthetic_budget_ready
        and deployment_budget_sweep.get("roi_measured_budget_replay_ready") is True
        and deployment_budget_sweep.get("roi_measured_categories", 0) >= 33
        and deployment_budget_sweep.get("max_roi_measured_budget_violation_rate", 1.0)
        <= 0.01
    )
    deployment_production_component_ready = (
        deployment_production.get("production_component_latency_ready") is True
        and deployment_production.get("categories", 0) >= 33
        and deployment_production.get("component_latency_protocol")
        == "per_image_synchronized_component_breakdown_v1"
    )
    deployment_production_budget_ready = (
        deployment_production_component_ready
        and deployment_production.get("multi_action_budget_sweep_ready") is True
        and deployment_production.get("max_budget_violation_rate", 1.0) <= 0.01
    )
    deployment_energy_ready = (
        deployment_production.get("energy_measurement_ready") is True
        or deployment_summary.get("energy_joules") is not None
    )
    deployment_cross_hardware_ready = (
        deployment_production.get("cross_hardware_ready") is True
        or int(deployment_production.get("hardware_profiles", 0) or 0) >= 2
    )
    deployment_ready = (
        deployment_readiness.get("production_deployment_ready") is True
        or (
            deployment_production_budget_ready
            and deployment_energy_ready
            and deployment_cross_hardware_ready
        )
    )
    deployment_smoke_ready = (
        deployment_readiness.get("smoke_protocol_ready") is True
        or (
            deployment_summary.get("evidence_level") == "synchronized_component_smoke_v1"
            and deployment_summary.get("latency_protocol") == "synchronized_batch_latency_v1"
            and all(
                key in deployment_summary
                for key in [
                    "latency_ms_p95",
                    "latency_ms_p99",
                    "budget_violation_rate",
                    "hardware",
                ]
            )
        )
    )
    if deployment_ready:
        deployment_missing_to_100 = "无缺口。"
    elif deployment_production_budget_ready:
        deployment_blockers = []
        if not deployment_cross_hardware_ready:
            deployment_blockers.append("跨硬件验证")
        if not deployment_energy_ready:
            deployment_blockers.append("能耗测量")
        deployment_blocker_text = "和".join(deployment_blockers) or "deployment readiness gate"
        deployment_missing_to_100 = (
            "真实 production component latency 和 10/25/50/75/100/150 ms "
            "六预算 LC-RDS sweep 已完成，required actions 全覆盖且预算违规率 <=1%；"
            f"仍缺{deployment_blocker_text}，不能写成完整 deployment release claim。"
        )
    elif deployment_production_component_ready:
        deployment_missing_to_100 = (
            "已有 synchronized batch=1 component smoke table、deployment readiness "
            "逐项 gate、measured synthetic LC-RDS action sweep、33 类 ROI budget "
            "replay，以及真实 inference component latency；已导出真实 production "
            "六预算 sweep 缺失命令表，并提供可恢复 runner 执行报告；仍缺执行"
            "剩余 runs、跨硬件和能耗测量。"
        )
    elif deployment_roi_measured_ready:
        deployment_missing_to_100 = (
            "已有 synchronized batch=1 component smoke table、deployment readiness "
            "逐项 gate、measured synthetic LC-RDS action sweep，以及基于实测 action "
            "p95 的 33 类 ROI budget replay；仍缺 production detector/verifier/"
            "scheduler/repair/IO 计时、真实 production budget violation、跨硬件和能耗测量。"
        )
    elif deployment_synthetic_budget_ready:
        deployment_missing_to_100 = (
            "已有 synchronized batch=1 component smoke table、deployment readiness "
            "逐项 gate 和 measured synthetic LC-RDS action sweep；仍缺 production "
            "detector/verifier/scheduler/repair/IO 计时、真实 LC-RDS measured "
            "budget violation、跨硬件和能耗测量。"
        )
    elif deployment_readiness.get("smoke_protocol_ready") is True:
        deployment_missing_to_100 = (
            "已有 synchronized batch=1 component smoke table 和 deployment readiness "
            "逐项 gate；仍缺 production detector/verifier/scheduler/repair/IO 计时、"
            "真实 LC-RDS measured budget violation、跨硬件和能耗测量。"
        )
    elif deployment_smoke_ready:
        deployment_missing_to_100 = (
            "已有 synchronized batch=1 component smoke table；仍缺 production "
            "detector/verifier/scheduler/repair/IO 计时、真实 LC-RDS measured "
            "budget violation、跨硬件、显存/能耗部署表。"
        )
    else:
        deployment_missing_to_100 = (
            "缺端到端 p95/p99、显存、能耗、跨硬件和预算违反率；旧混合 latency "
            "协议不能支撑强声明。"
        )
    rows.append(
        make_row(
            priority="P0",
            dimension="效率与部署",
            target_100=(
                "batch=1 端到端部署表，分 detector/verifier/scheduler/repair/IO 计时，"
                "含 p95/p99、显存、能耗、跨硬件和预算违规率。"
            ),
            status="complete" if deployment_ready else "partial",
            current_completion_percent=(
                100
                if deployment_ready
                else 96
                if deployment_production_budget_ready and deployment_energy_ready
                else 94
                if deployment_production_budget_ready
                else 88
                if deployment_production_component_ready
                else 84
                if deployment_roi_measured_ready
                else 82
                if deployment_synthetic_budget_ready
                else 80
                if deployment_readiness.get("smoke_protocol_ready") is True
                else 75
                if deployment_smoke_ready
                else 60
            ),
            current_evidence=evidence(
                "tests/test_latency_benchmark.py",
                "tables/deployment_latency/summary.json",
                "tables/deployment_latency/table_component_latency.csv",
                "tables/deployment_production_latency/summary.json",
                "tables/deployment_production_latency/table_production_component_latency.csv",
                "tables/deployment_production_latency/table_production_budget_replay.csv",
                "tables/deployment_production_latency/table_missing_production_budget_commands.csv",
                "tables/deployment_production_latency/energy_measurements/*.json",
                "tables/deployment_production_latency/production_budget_runner_summary.json",
                "tables/deployment_production_latency/table_production_budget_runner_report.csv",
                "tables/deployment_readiness/summary.json",
                "tables/deployment_readiness/table_deployment_readiness.csv",
                "tables/lc_rds_budget_audit/summary.json",
                "tables/lc_rds_budget_sweep/summary.json",
                "tables/lc_rds_budget_sweep/table_action_latency_summary.csv",
                "tables/lc_rds_budget_sweep/table_roi_measured_budget_sweep.csv",
                "tables/feature_first_fusion_aggregate_paper_package/table_efficiency_pareto.csv",
                "configs/scheduler/lc_rds_budget_v2.yaml",
            ),
            missing_to_100=deployment_missing_to_100,
            implementation_action=(
                "新增 deployment latency protocol；固定 warmup/measurements/CUDA sync "
                "并至少导出一套主硬件结果。"
            ),
            acceptance_criteria=(
                "形成部署表；LC-RDS 预算约束可复现；旧混合协议仅作历史参考。"
            ),
        )
    )

    engineering_ready = all(
        has_file(root, path)
        for path in [
            "requirements-lock.txt",
            "environment.yml",
            "Dockerfile",
            ".github/workflows/ci.yml",
            "CITATION.cff",
            ".zenodo.json",
        ]
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="代码工程",
            target_100=(
                "从本地工程变成可交付仓库：lockfile、environment、Docker、CI、citation、Zenodo "
                "metadata 和 artifact manifest 齐备。"
            ),
            status="complete" if engineering_ready else "partial",
            current_completion_percent=100 if engineering_ready else 70,
            current_evidence=evidence(
                "requirements-lock.txt",
                "environment.yml",
                "Dockerfile",
                ".github/workflows/ci.yml",
                "CITATION.cff",
                ".zenodo.json",
                "artifacts/manifest.json",
            ),
            missing_to_100=(
                "无默认 P0 缺口；每次提交前继续要求 pytest -q 干净通过。"
                if engineering_ready
                else "缺可交付工程文件或 CI 配置。"
            ),
            implementation_action=(
                "维持 release manifest；新增工具/表格后重新运行 manifest builder。"
            ),
            acceptance_criteria="第三方能安装、跑 smoke、重算至少一张主表；pytest -q 通过。",
        )
    )

    release_ready = prediction_release_ready(predictions, root)
    release_readiness = read_json(root / "tables/release_readiness/summary.json")
    local_release_ready = release_readiness.get("local_artifact_ready") is True
    public_release_ready = release_readiness.get("release_gate_passed") is True
    claim_traceability = read_json(root / "tables/claim_traceability/summary.json")
    claim_traceability_ready = claim_traceability.get("local_traceability_ready") is True
    local_release_with_trace_ready = local_release_ready and claim_traceability_ready
    rows.append(
        make_row(
            priority="P0",
            dimension="公开复现",
            target_100=(
                "论文数字可追溯到 checkpoint、33 类 prediction arrays、fixed threshold JSON、"
                "主表再生成脚本、SHA256 manifest、GitHub Release、Zenodo DOI 和 HF card。"
            ),
            status="complete" if public_release_ready else ("partial" if release_ready else "pending"),
            current_completion_percent=(
                100
                if public_release_ready
                else (94 if local_release_with_trace_ready else (90 if local_release_ready else (80 if release_ready else 35)))
            ),
            current_evidence=evidence(
                "tables/claim_traceability/summary.json",
                "tables/claim_traceability/table_claim_traceability.csv",
                "tables/release_readiness/summary.json",
                "tables/release_readiness/table_release_readiness.csv",
                "tables/release_readiness/github_release_notes.md",
                "artifacts/predictions_manifest.json",
                "artifacts/thresholds/synthetic_normal_fixed_threshold_v1.json",
                "artifacts/manifest.json",
                "REPRODUCE.md",
                "MODEL_CARD.md",
                "DATASETS.md",
            ),
            missing_to_100=(
                "无缺口。"
                if public_release_ready
                else "本地 release 工件和论文数字 traceability 已就绪；缺外部发布动作：GitHub Release、Zenodo DOI、HF model/dataset card 的实际链接/DOI。"
                if local_release_with_trace_ready
                else "本地 release 工件已就绪；缺 claim traceability 或外部发布动作：GitHub Release、Zenodo DOI、HF model/dataset card 的实际链接/DOI。"
                if local_release_ready
                else "缺外部发布动作：GitHub Release、Zenodo DOI、HF model/dataset card 的实际链接/DOI。"
                if release_ready
                else "缺 prediction arrays、threshold bundle 或 SHA256 manifest。"
            ),
            implementation_action=(
                "发布前重新生成 claim traceability 和 manifest；发布后把 DOI/HF/GitHub Release 链接写回 REPRODUCE/MODEL_CARD。"
            ),
            acceptance_criteria="每个论文数字能追到 config、seed、commit、artifact 和 SHA256。",
        )
    )

    submission_audit = read_json(root / "tables/submission_package_readiness/summary.json")
    submission_blockers = set(submission_audit.get("blocking_requirements", []))
    submission_local_ready = submission_audit.get("local_submission_draft_ready") is True or (
        has_file(root, "paper/manuscript.md")
        and has_reviewer_faq
        and has_statement_placeholders
    )
    submission_final_ready = submission_audit.get("final_upload_ready") is True
    submission_target_journal_ready = (
        submission_local_ready
        and "final_upload:target_journal" not in submission_blockers
    )
    rows.append(
        make_row(
            priority="P0",
            dimension="论文与投稿",
            target_100=(
                "目标期刊模板、补充材料、cover letter、声明页和审稿质疑预答复形成可上传包。"
            ),
            status=(
                "complete"
                if submission_final_ready
                else ("journal_gated" if submission_local_ready else "partial")
            ),
            current_completion_percent=(
                100
                if submission_final_ready
                else (86 if submission_target_journal_ready else (82 if submission_local_ready else 55))
            ),
            current_evidence=evidence(
                "paper/manuscript.md",
                "paper/supplement.md",
                "docs/reviewer_faq.md",
                "docs/submission_statement_placeholders.md",
                "docs/submission_reproducibility_checklist.md",
                "tables/submission_package_readiness/summary.json",
                "tables/submission_package_readiness/table_submission_readiness.csv",
            ),
            missing_to_100=(
                "无缺口。"
                if submission_final_ready
                else (
                    "结构化投稿草稿已齐，默认目标期刊 Pattern Recognition 已选；"
                    "仍缺模板转换、作者/单位/基金/利益冲突/数据代码声明的最终真实信息、"
                    "公开链接，以及全局 100% completion gate。"
                    if submission_target_journal_ready
                    else "结构化投稿草稿已齐；仍缺目标期刊选择、模板转换、作者/单位/基金/利益冲突/数据代码声明的最终真实信息、公开链接，以及全局 100% completion gate。"
                )
                if submission_local_ready
                else "缺 reviewer FAQ 或投稿声明页骨架。"
            ),
            implementation_action=(
                "运行 submission package readiness audit；按 Pattern Recognition/Elsevier 模板转换，填写真实作者、单位、基金和声明信息。"
            ),
            acceptance_criteria="形成可直接上传投稿系统的 manuscript、supplement、cover letter 和 statements。",
        )
    )

    diffusionad_compute = read_json(root / "tables/diffusionad_compute_plan/summary.json")
    rows.append(
        make_row(
            priority="P2",
            dimension="DiffusionAD full official 复现",
            target_100=(
                "只有目标期刊或审稿策略强要求时，按官方 15 类 3000 epoch 全量复现。"
            ),
            status="optional_not_required",
            current_completion_percent=100,
            current_evidence=evidence(
                "docs/diffusionad_reproduction_status_zh.md",
                "tables/diffusionad_compute_plan/summary.json",
                "baselines/official_sources.json",
            ),
            missing_to_100=(
                "默认 100% 不要求；若触发 P2，需要约 "
                f"{diffusionad_compute.get('estimated_total_gpu_hours', '3801')} GPUh 的 official run。"
            ),
            implementation_action=(
                "保持 source/assets/checkpoint provenance；不把 smoke/Lite 结果写成 official。"
            ),
            acceptance_criteria="官方配置、官方训练步数、15 类完整结果，或明确不纳入主论文。",
        )
    )

    return rows


def build_summary(rows: list[dict[str, Any]], date: str) -> dict[str, Any]:
    blockers = [
        row["dimension"]
        for row in rows
        if row.get("blocking_for_default_100")
    ]
    p0_rows = [row for row in rows if row["priority"] == "P0"]
    p1_rows = [row for row in rows if row["priority"] == "P1"]
    weighted = (
        sum(float(row["current_completion_percent"]) for row in p0_rows)
        / max(len(p0_rows), 1)
    )
    return {
        "schema": "lite-seer-ad-completion-gap-matrix-v1",
        "generated_on": date,
        "definition_of_100": (
            "paper story frozen, every retained contribution has auditable evidence, "
            "failed modules are honestly downgraded, statistics are fair, engineering "
            "is reproducible, and submission materials are complete"
        ),
        "default_mainline": (
            "Feature-first Lite-SEER-AD = label-free policy/threshold selection + "
            "HN-SEV false-positive suppression + LC-RDS budgeted local repair scheduling"
        ),
        "rows": len(rows),
        "p0_rows": len(p0_rows),
        "p1_rows": len(p1_rows),
        "p0_completion_percent_mean": round(weighted, 2),
        "default_100_ready": len(blockers) == 0,
        "blocking_p0_dimensions": blockers,
        "open_p1_dimensions": [
            row["dimension"]
            for row in p1_rows
            if row["status"] not in NON_BLOCKING_STATUSES
        ],
        "p2_policy": "CRV rescue and DiffusionAD full are optional conditional items.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "dimension",
        "target_100",
        "status",
        "current_completion_percent",
        "blocking_for_default_100",
        "current_evidence",
        "missing_to_100",
        "implementation_action",
        "acceptance_criteria",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lite-SEER-AD completion gap matrix",
        "",
        f"Generated on: {summary['generated_on']}",
        "",
        f"Default 100 ready: `{summary['default_100_ready']}`",
        f"P0 mean completion: `{summary['p0_completion_percent_mean']}%`",
        "",
        "Blocking P0 dimensions:",
        "",
    ]
    blockers = summary["blocking_p0_dimensions"]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "| Priority | Dimension | Status | Current % | Missing to 100 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        missing = str(row["missing_to_100"]).replace("\n", " ")
        lines.append(
            "| {priority} | {dimension} | {status} | {pct} | {missing} |".format(
                priority=row["priority"],
                dimension=row["dimension"],
                status=row["status"],
                pct=row["current_completion_percent"],
                missing=missing,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(root: Path, out_dir: Path, date: str) -> dict[str, Any]:
    rows = build_rows(root)
    summary = build_summary(rows, date)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "table_completion_gap_matrix.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(out_dir / "completion_gap_matrix.md", rows, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--date", default=DEFAULT_DATE)
    args = parser.parse_args()

    summary = write_outputs(args.root, args.out_dir, args.date)
    print(
        "Wrote completion gap matrix to "
        f"{args.out_dir} (default_100_ready={summary['default_100_ready']})"
    )


if __name__ == "__main__":
    main()
