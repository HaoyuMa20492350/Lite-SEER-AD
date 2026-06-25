# MVTec AD 2 官方上传状态

> 本数据集已从正文主实验和投稿验收门槛中移除。以下内容仅用于保留未来
> 可选附录资产；网页无法登录不会阻塞当前论文投稿。

## 当前状态

核对日期：2026-06-14。

- 官方入口：`https://benchmark.mvtec.com/`
- 本地 public：24/24 运行完成。
- private + private-mixed：4090/4090 图像完成。
- 推荐上传包：
  `submissions/mvtec_ad2_seed7_model256.tar.gz`
- 大小：`358262816` bytes，约 `341.67 MiB`。
- SHA256：
  `9ae82e41018ad17a59c5f9ae176f4a12355cef9f7bc9bc143e342145deff266b`
- 归档成员：8180，单一顶层目录。
- TIFF：4090 个，`float16`，`256 x 256`。
- PNG：4090 个，单通道二值图，`256 x 256`。
- 官方 checker：passed。

模型输出分辨率上传符合官方说明；服务器会将结果双线性上采样到原图尺寸。
该包相对原图尺寸包减少 `96.03%`，不改变模型原始 256 像素输出或冻结阈值。

## 外部阻塞

官方上传页需要登录。当前自动化环境没有用户账号、组织/高校邮箱登录会话或
Bearer token，无法代表用户启动上传。官方页面还提示：开始上传即占用一次
提交配额，评测可能持续数小时。

## 可选的未来动作

收到官方结果导出后运行：

```powershell
python tools/import_mvtec_ad2_official_results.py `
  --input "<SERVER_RESULT.json>" `
  --source-url "https://benchmark.mvtec.com/" `
  --submission-id "<SUBMISSION_ID>" `
  --evaluated-at "<ISO_DATETIME>" `
  --update-draft
```

保存同日 leaderboard JSON 后运行：

```powershell
python tools/audit_mvtec_ad2_leaderboard.py `
  --input "<LEADERBOARD.json>" `
  --update-evaluation
```

当前论文不包含 AD2 主结果。即使未来完成上传，也只有在服务器结果与同期
leaderboard 审计均完成后，才可在附录中加入相应 private 指标；不得因此
改写已经冻结的三数据集主结论。
