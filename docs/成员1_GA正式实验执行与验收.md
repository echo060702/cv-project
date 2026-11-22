# 成员1 G/A 正式实验执行与验收（已完成）

状态（2026-11-22 16:28）：G/A 六次全部跑满 20 epoch，六套 checkpoint 已重载重放 1,200 张 dev 并与日志一致，正式汇总已生成。14:27 的"停止并中途交接"版本作废。

## 执行记录

- 串行执行 G17→A17→G29→A29→G43→A43，输出 `runs/baselines`；G43 从第 6 轮 checkpoint 续跑至 20，A43 新建跑满 20。
- 旧 `baseline_runs` 是更早的中止历史，不复用、不覆盖、不随包交付。
- 冻结配方未改：20 epoch / 224 / width 32 / batch 16 / workers 0 / 种子 17,29,43；`configs/*.json` 与 `validation/budget_freeze.json` 哈希一致。
- checkpoint 按干净 dev 单 patch Accuracy→较低 BCE→较早 epoch 选择，阈值 0.5。

## 验收

- [x] 六次 20 epoch 日志连续、数值有限，config/metadata/environment/metrics/best/last/1,200 行 dev 预测齐全。
- [x] 六套 best.pt 重载重放，逐图预测与记录一致：`results/baselines/baseline_summary.json` passed=true、`checkpoint_predictions_replayed=true`。
- [x] 三种子均值/标准差/逐 run 表/训练曲线/成本：`summary.csv`、`per_run.csv`、`training_curves.png/pdf`。
- [x] 分析稿：`docs/G_A正式基线结果与分析.md`，明确 dev 复用、视野与尺度混淆、格式偏差三条边界。
- [x] 未评价 holdout、未解码 test 像素（`holdout_evaluated=false`、`test_pixels_decoded=false`）。
- [x] 第一轮 `validation/baseline_delivery_round1.json`、第二轮 ZIP 旁核验报告，均绑定实际产物。

## 结果

G 0.8047 ± 0.0005、A 0.8806 ± 0.0260（A−G = +7.58 个百分点）。这是 dev 选择后的成绩，不是独立泛化结论。
