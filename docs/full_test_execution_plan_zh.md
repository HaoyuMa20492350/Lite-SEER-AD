# Lite-SEER-AD 完整测试集执行方案

## 当前协议

- MVTec AD：15 类，1725 张官方测试图。
- VisA：12 类，严格使用 `split_csv/1cls.csv`，8659 张训练图、2162 张测试图。
- MPDD：6 类，458 张官方测试图。
- 合计：33 类、4345 张测试图。
- DTD：不作为测试集；HN-SEV 使用完整 5640 张纹理库。
- 每类运行 9 个设置，共 297 个类别设置和 39105 条图像级记录。

## 模型策略

- MVTec AD、MPDD：复用已有扩散模型和特征先验。
- MVTec AD、MPDD：重新挖掘完整训练集 hard negatives，并使用完整 DTD 重训 HN-SEV。
- VisA：由于旧加载器存在官方训练/测试混用，扩散模型、特征先验和 HN-SEV 全部重训。
- CRV 权重固定为 `0.35`，完整测试集上不再搜索权重，避免测试标签调参。

## 执行

仅查看完整命令：

```powershell
python tools/run_full_test_suite.py --dry-run --resume
```

正式执行并支持断点续跑：

```powershell
python tools/run_full_test_suite.py --resume
```

执行顺序：

1. `tools/audit_dataset_protocol.py`
2. 为 MVTec AD、MPDD 复制可复用基础检查点
3. 重建 MVTec AD、MPDD 的 HN-SEV
4. 完整重训 VisA
5. 对 33 类执行 9 个设置的完整测试集推理
6. `tools/audit_full_test_coverage.py`
7. `tools/export_full_test_paper_tables.py`

## 完成判定

只有 `tables/full_test_coverage_audit.json` 中 `complete=true` 才表示完整实验完成。审计同时要求：

- 每个预测文件的图片路径与官方测试路径完全一致；
- 无缺失、无额外图片、无重复图片；
- 所有预测数组长度一致；
- VisA 基础模型使用完整官方训练集；
- 三个数据集的 HN-SEV 均使用完整训练集和 5640 张 DTD；
- 总计 297 个设置全部完成。

当前只完成了协议修复、运行入口和审计工具，尚未启动长时间训练或全量推理。
