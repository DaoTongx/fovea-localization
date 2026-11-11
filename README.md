# 眼底图像黄斑中央凹定位（Fovea Localization）

机器学习课程设计 · 任务三。给定一张眼底图像，预测黄斑中央凹在原图上的像素坐标 `(X, Y)`；
中央凹不可见时输出 `(0, 0)`。评价指标为把 X / Y 拆成两行后的 RMSE。

Kaggle 公开榜成绩：**RMSE 14**（最终提交版本：640 输入 + 5 折 × 3 seed 集成）。
常数基线（训练集坐标均值）为 146.6 px。

---

## 方法概要

```
眼底图 → letterbox(640×640) → ResNet18 → 1/4 分辨率热力图 → 峰值窗口质心解码
       → 逆变换回原图像素 → 水平翻转 TTA + 15 个模型热力图平均 → submission.csv
```

| 环节 | 做法 | 动机 |
|---|---|---|
| 坐标表示 | letterbox 等比缩放 + 补黑边，坐标随同一仿射变换 | 两种分辨率统一语义；误差在原图尺度计算 |
| 监督方式 | 高斯热力图 + BCE-with-logits | 密集监督；避免 MSE 把热力图压平 |
| 解码 | 峰值 + 7×7 窗口加权质心 | 避免全图 soft-argmax 被背景拉向图像中心 |
| 稳定性 | last-conv bias = −4、5 epoch warmup、失败折自动重训 | 解决热力图饱和导致的梯度消失 |
| 集成 | 5 折 × 3 seed，热力图层面平均 | 抑制单次训练崩溃 |

## 主要结果

| 版本 | 关键改动 | OOF RMSE | 榜单 RMSE |
|---|---|---|---|
| v1 | 直接回归 + MSE 热力图 | 52.32 | 46 |
| v2 | BCE + bias(−4) + warmup + 失败折重训 | — | 18 |
| v3 | 512 输入 + 3 seed 集成 | 16.90 | 16 |
| v4 | 120 epoch + 增强放大 | 19.77 | 未提交 |
| v5 | v3 + 绿通道 CLAHE | 18.82 | 未提交 |
| v6 | v3 + 640 输入（最终） | 23.00 | **14** |

v4 / v5 / v6 的逐图误差明细保存在 `results/` 下，可直接复算报告中的表格与图。

## 目录结构

```
src/          fovea_localization.py   主流程（与 notebook 内容一致）
notebooks/    fovea_localization.ipynb Kaggle 上直接运行
report/       实验报告（14 页完整版 / 6 页精简版）+ 构建脚本 + 全部插图
logs/         v1（崩溃）与 v3（修复后）两次训练日志
results/      三次运行的逐图 OOF 明细与提交文件
tools/        analyze.py  误差归因分析
```

## 环境

```bash
pip install -r requirements.txt
```

PyTorch 2.x + torchvision（GPU 建议）、OpenCV、pandas、numpy、python-docx。

## 数据

比赛数据（`detection/`）不随仓库分发，请从课程平台下载后放到：

```
raw/detection/
├── train/  test/  train_location/
├── fovea_localization_train_GT.csv
└── sample_submission.csv
```

## 复现

**在 Kaggle 上**：把 `notebooks/fovea_localization.ipynb` 导入，挂载数据（Dataset 结构保持 `detection/` 作为根），
开启 GPU 与 Internet，Run All。默认配置 640 输入 / 3 seed，约 50 分钟。

**在本地**：

```bash
export FOVEA_DATA=/path/to/raw/detection
python src/fovea_localization.py
```

关键超参在文件开头的 `CFG` 中；也可用环境变量覆盖：`IMG`、`EPOCHS`、`FOLDS`、`SEEDS`、`MODES`、`CLAHE`。

**误差归因**：

```bash
python tools/analyze.py results/v6_640_3seed/oof_detail.csv
```

**重新生成报告**（需要数据与上述 results）：

```bash
python report/build_report_task3.py     # 完整版（14 页，9 图 6 表）
python report/build_report_compact.py   # 精简版（6 页，4 图 3 表）
```

## 说明

- 训练集 80 张全部可见，没有 `(0,0)` 负样本，因此模型不做可见性判别，一律输出坐标。
- 详细的方法设计、消融与失败分析见 `report/任务三_眼底黄斑定位_实验报告.docx`。
