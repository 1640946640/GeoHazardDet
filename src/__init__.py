# ============================================================
#  GeoHazardDet - 地质灾害识别项目
#  Debris Flow & Landslide Detection Project
# ============================================================
#
# 【项目概述】
# 基于YOLOv8的泥石流/滑坡地质灾害实时识别系统
# 适用于地质灾害监测、风险预警等应用场景
#
# 【核心功能】
# - 泥石流、滑坡两类目标检测
# - 实时推理与批量推理支持
# - 模型训练、评估、导出完整流程
# - 支持GPU/CPU推理
#
# 【目录结构】
# GeoHazardDet/
# ├── datasets/           # 数据集目录
# │   ├── images/         # 图片目录（train/val/test）
# │   └── labels/         # 标注目录（train/val/test）
# ├── src/                # 源代码
# │   ├── data/           # 数据处理模块
# │   ├── models/         # 模型模块
# │   └── utils/          # 工具模块
# ├── configs/            # 配置文件
# └── runs/               # 训练输出
#
# 【快速开始】
# 1. 安装依赖: pip install -r requirements.txt
# 2. 准备数据: 运行 src/data/converter.py 转换数据
# 3. 开始训练: python src/train.py
# 4. 模型推理: python src/predict.py --weights runs/train/exp/weights/best.pt
#
# 【作者】
# GeoHazardDet Team
#
# 【创建日期】
# 2026-02-10
#
# ============================================================
