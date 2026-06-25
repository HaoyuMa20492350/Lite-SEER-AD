# Lite-SEER-AD 完整测试集实验完成报告

## 覆盖结论

- 数据集：MVTec AD、VisA、MPDD。
- 类别：33/33。
- 官方测试图像：4345/4345。
- 实验设置：9。
- 完整运行：297/297。
- 图像级实验记录：39105。
- VisA 使用官方 `split_csv/1cls.csv`，训练集与测试集重叠为 0。
- HN-SEV 使用完整 DTD 纹理库，共 5640 张图像。

## 阈值协议

每个类别使用官方训练正常图和确定性合成缺陷，在 seed
`7, 13, 23` 上冻结一套像素阈值。校准不读取真实测试异常标签或
真实测试异常掩码。33 个类别的 9 个设置具有逐元素相同的检测热图，
因此同一类别共享同一套固定阈值。297 个运行均通过
`synthetic_normal_fixed_threshold_v1` 协议审计。

## 主方法结果

以下为 `feature_tuned_crv` 的类别宏平均：

| 数据集 | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | 固定 Dice |
|---|---:|---:|---:|---:|---:|
| MVTec AD | 0.8920 | 0.9473 | 0.8055 | 0.3914 | 0.3986 |
| VisA | 0.8111 | 0.9631 | 0.7754 | 0.2948 | 0.3038 |
| MPDD | 0.7840 | 0.9564 | 0.8366 | 0.2062 | 0.2276 |
| 33 类宏平均 | 0.8430 | 0.9547 | 0.8002 | 0.3226 | 0.3330 |

33 类 oracle Dice 宏平均为 0.3765，固定阈值 Dice 为 0.3330，
平均差距为 0.0435。oracle 数值只能作为阈值上界，不能作为正式主结果。

## 结果边界

当前结果证明完整官方测试覆盖和无测试真值泄漏的固定阈值评估已经完成，
但不能据此宣称通用 SOTA。固定 Dice 最弱的类别包括 MPDD
`bracket_white`（0.0000）、VisA `macaroni2`（0.0165）、MPDD
`bracket_black`（0.0202）和 MVTec `screw`（0.0350）。后续优化应优先
处理小缺陷、低 Pixel AP 类别和跨类别阈值泛化，而不是继续扩大测试覆盖。

## 权威证据

- 覆盖审计：`tables/full_test_coverage_audit.json`
- 阈值冻结报告：`tables/full_test_threshold_freeze_report.json`
- 论文表格包：`tables/fulltest_paper_package/`
- 数据协议审计：`tables/full_test_protocol_audit.json`
