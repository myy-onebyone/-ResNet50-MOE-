# 眼底疾病智能诊断系统

数字图像处理期末综合报告，课题方向为医学图像处理。

## 运行

```bash
python main_window_compare.py
```

上传左右眼眼底图 → 智能诊断 → MATLAB 图像量化分析。

## 主要文件

| 文件 | 说明 |
|------|------|
| `main_window_compare.py` | PyQt5 主界面，双模型对比 + 集成诊断 + 量化分析 |
| `mymodel.py` | ResNet50 多标签分类模型，ImageNet 预训练 |
| `MoE_model.py` | Mixture of Experts 双流门控融合模型 |
| `dataloader.py` | ODIR-5K 数据集加载，过采样，数据增强 |
| `train.py` / `train_rigorous.py` | 训练脚本 |
| `compare_experiment.py` | ResNet50 vs MoE 对比实验 |
| `fundus_preprocess.py` | 眼底图像预处理，CLAHE / 高斯滤波 / 组合流水线 |
| `preprocess_analysis.py` | 预处理量化分析，生成对比图与报告 |
| `matlab_style_analysis.py` | 原始图像 vs 组合流水线 增强特征对比 |
| `test_final.py` / `assess.py` | 测试与评估 |

## 模型

- **ResNet50**: 左右眼独立推理，Sigmoid 概率取 max 融合
- **MoE**: 共享 ResNet50 提取双眼特征，拼接后经门控网络分配权重给 8 个专家子网络，Focal Loss 训练

## 图像预处理

三种方法对比：

- **CLAHE** — LAB 空间局部直方图均衡化，EME 增益约 2 倍
- **高斯滤波** — 5×5 核去噪，PSNR ~42dB，保留结构
- **完整流水线** — CLAHE → 高斯 → 反锐化掩膜，综合增强效果最佳

量化指标覆盖：灰度统计 (均值/标准差/偏度/峰度/信息熵)、RGB 直方图、Sobel 梯度、FFT 频谱、频段能量分布。

## 数据集

Kaggle ODIR-5K，约 3500 患者 7000 张眼底图，8 类标签（正常/糖尿病/青光眼/白内障/AMD/高血压/近视/其他异常）。

▪ https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k

