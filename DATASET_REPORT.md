# GeoHazardDet - 泥石流/滑坡地质灾害识别项目

## 📊 数据集分析报告

### 数据集结构

| 目录              | 文件数  | 说明                               |
| ----------------- | ------- | ---------------------------------- |
| `datasets/img/`   | 1635 张 | 无人机航拍影像 (TIFF格式, 512×512) |
| `datasets/label/` | 1635 个 | 分割掩码标签 (TIFF格式)            |
| `datasets/mask/`  | 1635 个 | 辅助掩码文件                       |

### 标签格式分析

**原始标签格式**: 分割掩码 (Segmentation Mask)

- 格式: TIFF图片 (512×512 RGB)
- 像素值: 0=背景, 255=目标区域
- 特点: 语义分割格式, 需转换为目标检测框

**转换后格式**: YOLO边界框

```
<class_id> <x_center> <y_center> <width> <height>
```

示例: `0 0.499023 0.499023 0.998047 0.998047`

### 类别映射

| 原始值 | 类别ID | 类别名称    | 说明   |
| ------ | ------ | ----------- | ------ |
| 255    | 0      | debris_flow | 泥石流 |
| 其他   | 1      | landslide   | 滑坡   |

---

## ✅ 数据集验证结果

### 样本可视化

已生成以下可视化图片 (`runs/visualize/`):

1. **`*_original.jpg`** - 原始图片
2. **`*_mask.jpg`** - 分割掩码 (红色=目标)
3. **`*_overlaid.jpg`** - 掩码叠加效果
4. **`*_annotated.jpg`** - 边界框标注

### YOLO标签验证

```
文件: datasets/labels_yolo/moxizheng_0.2m_UAV0001.txt
内容: 0 0.499023 0.499023 0.998047 0.998047
格式: ✓ 正确 (class_id x_center y_center width height)
```

---

## 🚀 下一步操作

### 1. 完整数据集转换（推荐）

```bash
# 将所有分割掩码转换为YOLO格式，并划分数据集
python src/data/prepare_dataset.py
```

### 2. 开始训练

```bash
# 基础训练
python src/train.py --data configs/disaster.yaml --epochs 100

# 自定义参数训练
python src/train.py \
  --data configs/disaster.yaml \
  --epochs 200 \
  --batch 16 \
  --model yolov8n.pt \
  --device 0
```

### 3. 模型推理

```bash
# 单张图片检测
python src/predict.py \
  --weights runs/train/exp/weights/best.pt \
  --source test.jpg

# 批量检测
python src/predict.py \
  --weights runs/train/exp/weights/best.pt \
  --source datasets/images/val/
```

---

## 📁 生成的文件

### 配置文件

- `configs/disaster.yaml` - YOLO数据集配置

### 数据处理脚本

- `src/data/prepare_dataset.py` - 完整数据集准备脚本
- `src/data/analyze_dataset.py` - 数据集分析脚本
- `src/data/visualize_labels.py` - 标注可视化脚本

### 训练/推理脚本

- `src/train.py` - 模型训练
- `src/predict.py` - 模型推理
- `src/evaluate.py` - 模型评估

### 输出目录

- `runs/visualize/` - 可视化结果
- `runs/train/` - 训练输出
- `runs/predict/` - 推理输出

---

## 📝 注意事项

1. **标签格式**: 当前标签是分割掩码，已转换为YOLO边界框格式
2. **图片格式**: 原始TIFF已转换为JPEG用于训练
3. **类别**: 共2类 (泥石流=0, 滑坡=1)
4. **样本数**: 1635对图片-标签

---

**生成时间**: 2026-02-12
**项目路径**: `C:\Users\16409\Desktop\ProgrammingOrders\GeoHazardDet`
