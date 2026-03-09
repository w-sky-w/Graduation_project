# Fingerflex ECoG Baseline & Robustness

本仓库提供针对 Stanford `##_fingerflex.mat` 数据格式的两个部分：

1. `notebooks/fingerflex_visualization.ipynb`：可视化 notebook（电极时域、手指轨迹、cue、相关性、功率谱）。
2. baseline 与电极缺失鲁棒性实验代码：
   - `src/fingerflex_baseline.py`
   - `scripts/dropout_experiment.py`

## 数据格式假设

- `data`: `(time, channels)`，采样率 1000Hz
- `flex`: `(time, 5)`，对应 `thumb/index/middle/ring/little`
- `cue`: `(time, 1)`，编码 `0-5`

## 快速开始

```bash
python scripts/dropout_experiment.py \
  --mat /your/path/01_fingerflex.mat \
  --dropouts 0 0.05 0.10 0.15 0.20 \
  --repeats 3 \
  --out-csv results/dropout_metrics.csv \
  --out-plot results/dropout_curve.png
```

## Baseline 方法

- 使用时间滞后特征（默认 300ms 窗口，每 50ms 一个 lag）。
- 模型：`MultiOutputRegressor(Ridge)` 同时预测 5 根手指 flex。
- 时间切分：70% 训练，15% 验证，15% 测试（当前主结果使用测试集）。
- 电极缺失模拟：随机选择一定比例电极，将该电极在所有 lag 特征置零。

输出指标包括：

- `corr_mean` 与各手指相关系数
- `r2_mean` 与各手指 `R^2`

