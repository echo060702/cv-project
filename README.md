> **2026-11-25 交付分支状态**：已纳入成员2 B/C/D交付与成员3 E/F全部六次、120轮结果。
> E/F权重和结果位于 `runs/member3/`、`results/member3/`；权重通过 Release 附件提供，不进入git。
> 当前复现入口、环境差异及证据位置见 [成员3交付与复现](docs/成员3_交付与复现.md)。
> 本地汇总和32项发布自测通过；历史CUDA重放报告与原自动汇总失败日志均保留。
> 以下为成员1的历史交接说明，原离线包的清单验证命令不适用于当前整合工作区。

> **成员1历史 GitHub 仓库版说明**：当时仓库只含代码、配置、文档、审计清单与 G/A 六次实验的日志/结果（约 9 MB）。
> 官方图像与 checkpoint 不进入 git（单文件超 GitHub 100 MB 限制）。
> 权重从 [Release v1.0-weights](https://github.com/echo060702/cv-project/releases/tag/v1.0-weights) 下载 `checkpoints.zip`，
> 图像从官方 Hugging Face 数据集获取；步骤与 SHA256 见 `docs/仓库说明_数据与权重获取.md`。
>
> 成员2（B/C/D）、成员3（E/F）跑完后如何提交与发版权重：见 `docs/队友交付流程_发布权重.md`。
> 团队交接用的完整离线包（含 8,000 训练图与权重，2.28 GiB）另行分发，不在本仓库内。

# AI 生成图像识别：数据 + G/A 六次正式基线（交接包）

**从 `00_交接说明_两位成员先读.md` 开始；结论看 `docs/G_A正式基线结果与分析.md`。**

成员1（用户）已完成：官方数据下载与审计、固定划分、配对数据加载、**G/A 各 3 种子共 6 次 20-epoch 正式基线**、结果核验与汇总、数据与基线章节素材。再往后：成员2 做 B/C/D 九次与最终模型；成员3 做 E/F 六次、独立评估、CSV 与论文/展示整合。**G/A 已跑完，不要重跑。**

六次成绩（dev 单 patch，阈值 0.5，按 dev Accuracy→低 BCE→早 epoch 选 checkpoint）：G 0.8047 ± 0.0005，A 0.8806 ± 0.0260，A−G = +7.58 个百分点。汇总在 `results/baselines/`，六套 run 在 `runs/baselines/`。

## 解压、校验与安装

完整解压，进入内层 `ai_detection_team_20261122` 目录，改动前执行：

```powershell
python -X utf8 -m tools.package_team --verify .
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m tools.verify_data_handoff --output runs/received_data_check.json
```

本机环境：Python 3.12.7、PyTorch 2.5.1 + CUDA 11.8、NumPy 2.4.4、Pillow 12.2.0、Matplotlib 3.11.1、RTX 4060 Laptop 8 GB。新机器按实际驱动安装，示例（CUDA 11.8）：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

无 NVIDIA GPU 时安装 CPU 版 torch 只能跑自测与数据开发，正式训练需 GPU 环境。

## 包内结构

| 路径 | 内容 |
|---|---|
| `00_交接说明_两位成员先读.md` | 当前职责、完成事实、运行顺序、验收 |
| `要求.md`、`三人小组分工_待审核.md` | 课程原文与当前分工 |
| `results/baselines/` | 六次汇总、逐 run 表、训练曲线、checkpoint 重放证据 |
| `runs/baselines/` | 六套 20-epoch run（日志、best/last、dev 逐图预测、environment） |
| `data/` | 8,000 训练图、1,000 隔离无标签 test、官方标签与模板、固定划分、train 统计、审计 |
| `team_data.py`、`tools/` | 配对数据加载与准备/核验/汇总/打包工具 |
| `cvdetect/`、`configs/` | 共享模型工程与七组冻结配置 |
| `docs/` | 分工、接手指南、数据审计、结果分析、环境与坑 |
| `tests/`、`validation/` | 21 项自测、数据/来源/计时/预算/核验证据 |

## 数据与方法规则

禁止预训练，只能随机初始化；训练只消费冻结 train/dev，holdout 确认前冻结方法，官方 test 仅最终推理。JPEG 在整幅变换后编码，再对两视图对齐裁剪；模型内部归一化与残差，外部不重复处理。真实图全 JPEG、AI 图全 PNG，813 张短边不足 224——格式/尺寸偏差必须进分析，随机划分成绩不支持跨来源泛化。

共同预算：20 epoch、224、width 32、batch 16、workers 0、种子 17/29/43，七组同配方，哈希见 `validation/budget_freeze.json`。

## 当前导出

`dist/AI图像识别_数据与GA六次基线_完整交接包_20261122.zip`（含全部数据、六次 run、汇总与核验报告）。旧包（100/105/9137 文件、4 次中途版）均为历史，不再分发。本包不含 Python 环境、GPU 驱动、B/C/D/E/F 结果、最终模型、提交 CSV、论文 PDF 与 `team_eval.py`。

## 已知问题

临时目录 ACL：沙箱下 `tempfile.mkdtemp` 的 0o700 目录会触发 `PermissionError: [WinError 5]`，`tests/__init__.py` 与 `tools/__init__.py` 已内置兼容层，**不要删除**；必要时把 `TMP`/`TEMP` 指到普通可写目录。详见 `docs/环境与已知问题.md`。
