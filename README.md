# YOLOv8 安全帽目标检测

基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 的安全帽佩戴检测项目，能够识别施工现场图像中的**安全帽（helmet）**、**头部（head）**和**人员（person）** 三类目标。

## 功能特性

- **3 类目标检测**：安全帽（helmet）、头部（head）、人员（person）
- **YOLOv8n 模型**：轻量高效，适合快速推理
- **自定义标注工具**：`labelImgse.py` 用于补充标注 person 类别
- **完整训练流程**：Jupyter Notebook 一键训练与评估

## 数据集

| 数据集  | 图片数量 |
| ------- | -------- |
| 训练集  | 4,738    |
| 验证集  | 1,000    |

数据来源为 Hard Hat Workers 数据集，原始标注为 Pascal VOC XML 格式，已转换为 YOLO 格式。

## 项目结构

```
hw3/
├── YOLOv8 目标检测.ipynb   # 训练与检测 Notebook
├── data.yaml               # 数据集配置文件
├── labelImgse.py           # 自定义标注工具（person 补标）
├── dataset/
│   ├── train/
│   │   ├── images/         # 训练集图片（4,738张）
│   │   └── labels/         # 训练集标签（YOLO 格式）
│   └── val/
│       ├── images/         # 验证集图片（1,000张）
│       └── labels/         # 验证集标签
└── helmet/                 # 原始数据（图片 + Pascal VOC XML）
    ├── images/
    ├── labels/
    ├── annotations/
    └── classes.txt
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8

```bash
pip install ultralytics torch matplotlib pillow
```

### 训练模型

打开 Jupyter Notebook 运行完整训练流程：

```bash
jupyter notebook "YOLOv8 目标检测.ipynb"
```

或直接使用命令行训练：

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=100, imgsz=640)
```

### 推理检测

```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')
results = model.predict(source='your_image.jpg', save=True)
```

## 标注工具

`labelImgse.py` 是一个基于 Matplotlib 的轻量标注工具，用于在已有 helmet/head 标注的基础上补充 person 类别标注：

```bash
python labelImgse.py
```

- 鼠标拖拽绘制标注框
- 键盘快捷键切换图片、保存标注

## 类别说明

| 类别 ID | 名称   | 说明           |
| ------- | ------ | -------------- |
| 0       | helmet | 佩戴安全帽的人 |
| 1       | head   | 未佩戴安全帽   |
| 2       | person | 人员           |

## 模型权重

模型权重文件（`.pt`）未包含在仓库中，请通过以下方式获取：

- 预训练权重：`yolov8n.pt` 会在首次训练时自动从 Ultralytics 下载
- 训练后权重：运行 Notebook 训练后在 `runs/detect/train/weights/` 目录生成
