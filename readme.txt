# 17组-基于ResNet50的眼底医学影像多标签诊断

## 环境要求
- Python 3.8+
- PyTorch, torchvision
- PyQt5, Pillow, OpenCV, numpy, matplotlib, scipy
- 安装: `pip install torch torchvision PyQt5 Pillow opencv-python numpy matplotlib scipy`

## 运行方式
```bash
cd src
python main_window_compare.py


## 文件夹说明
-src/      — 全部Python源码（主程序 main_window_compare.py双击运行）
- models/   — 训练好的模型权重（ResNet50 + MoE）
- data/     — 样本眼底图像数据
- output/   — 分析结果图表
- docs/     — 项目文档和训练日志
