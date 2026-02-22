# -*- coding: utf-8 -*-
"""
============================================================
完整数据集准备脚本
将分割掩码转换为YOLO格式并生成训练配置
============================================================

【功能】
1. 分析数据集结构
2. 将分割掩码转换为YOLO边界框标注
3. 划分训练集/验证集/测试集
4. 生成配置文件

【使用】
python src/data/prepare_dataset.py --convert --split --samples 1635

============================================================
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image


class DatasetPreparer:
    """数据集准备器"""
    
    # 类别映射
    CLASS_NAMES = {
        0: 'debris_flow',   # 泥石流
        1: 'landslide',     # 滑坡
    }
    
    def __init__(
        self,
        img_dir: str = 'datasets/img/',
        label_dir: str = 'datasets/label/',
        output_dir: str = 'datasets/',
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        初始化数据集准备器
        
        Args:
            img_dir: 原始图片目录
            label_dir: 原始标签目录（分割掩码）
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        
    def analyze_original_data(self) -> Dict:
        """分析原始数据集"""
        print("=" * 60)
        print("原始数据分析")
        print("=" * 60)
        
        # 获取文件列表
        img_files = list(self.img_dir.glob('*.tif'))
        label_files = list(self.label_dir.glob('*.tif'))
        
        print(f"\n[文件统计]")
        print(f"  图片文件: {len(img_files)} 个")
        print(f"  标签文件: {len(label_files)} 个 检查配对
        img_stems")
        
        # = set(f.stem for f in img_files)
        label_stems = set(f.stem for f in label_files)
        matched = img_stems & label_stems
        
        print(f"  配对文件: {len(matched)} 对")
        
        # 分析一个样本
        if img_files:
            sample_img = Image.open(img_files[0])
            sample_label = Image.open(label_files[0])
            
            print(f"\n[样本分析]")
            print(f"  图片尺寸: {sample_img.size}")
            print(f"  图片模式: {sample_img.mode}")
            
            label_array = np.array(sample_label)
            if len(label_array.shape) == 3:
                label_array = label_array[:, :, 0]
            unique_vals = np.unique(label_array)
            
            print(f"  标签唯一值: {unique_vals}")
            
            for v in unique_vals:
                count = (label_array == v).sum()
                print(f"    值={v}: {count} 像素 ({count/label_array.size*100:.2f}%)")
        
        return {
            'total_samples': len(matched),
            'img_files': img_files,
            'label_files': label_files,
        }
    
    def convert_mask_to_yolo(self, limit: int = None) -> Tuple[List[Path], Dict]:
        """
        将分割掩码转换为YOLO格式标签
        
        Args:
            limit: 限制转换数量（None表示全部）
            
        Returns:
            (转换的文件列表, 统计信息)
        """
        print("\n" + "=" * 60)
        print("转换分割掩码为YOLO边界框")
        print("=" * 60)
        
        # 创建输出目录
        yolo_label_dir = self.output_dir / 'labels_yolo'
        yolo_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取标签文件
        label_files = list(self.label_dir.glob('*.tif'))
        
        if limit:
            label_files = label_files[:limit]
        
        print(f"\n转换 {len(label_files)} 个文件...")
        
        stats = {
            'total': len(label_files),
            'success': 0,
            'failed': 0,
            'class_counts': defaultdict(int),
        }
        
        converted_files = []
        
        for label_path in label_files:
            stem = label_path.stem
            
            try:
                # 读取掩码
                mask = np.array(Image.open(label_path))
                
                # 转灰度
                if len(mask.shape) == 3:
                    mask_gray = mask[:, :, 0]
                else:
                    mask_gray = mask
                
                # 找到所有非零区域
                unique_vals = np.unique(mask_gray)
                
                boxes = []
                for val in unique_vals:
                    if val == 0:
                        continue
                    
                    # 找到该值的像素
                    ys, xs = np.where(mask_gray == val)
                    
                    if len(xs) == 0:
                        continue
                    
                    # 计算边界框 (归一化)
                    img_h, img_w = mask_gray.shape
                    
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    
                    x_center = (x1 + x2) / 2 / img_w
                    y_center = (y1 + y2) / 2 / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    
                    # 类别ID: 255=泥石流(0), 其他=滑坡(1)
                    class_id = 0 if val == 255 else 1
                    
                    boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    stats['class_counts'][class_id] += 1
                
                # 保存YOLO格式标签
                if boxes:
                    output_path = yolo_label_dir / f'{stem}.txt'
                    with open(output_path, 'w') as f:
                        f.write('\n'.join(boxes))
                    
                    converted_files.append((stem, output_path))
                    stats['success'] += 1
                    
            except Exception as e:
                print(f"  错误: {stem} - {e}")
                stats['failed'] += 1
        
        print(f"\n转换完成!")
        print(f"  成功: {stats['success']}")
        print(f"  失败: {stats['failed']}")
        print(f"  类别统计:")
        for cls_id, count in stats['class_counts'].items():
            name = self.CLASS_NAMES.get(cls_id, f'Class_{cls_id}')
            print(f"    {name}({cls_id}): {count}")
        
        return converted_files, stats
    
    def split_dataset(
        self,
        converted_files: List[Tuple[str, Path]],
    ) -> Dict[str, List[str]]:
        """
        划分数据集
        
        Args:
            converted_files: 转换后的文件列表 [(stem, path), ...]
            
        Returns:
            划分结果 {'train': [...], 'val': [...], 'test': [...]}
        """
        print("\n" + "=" * 60)
        print("划分数据集")
        print("=" * 60)
        
        # 打乱文件列表
        random.shuffle(converted_files)
        
        total = len(converted_files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        splits = {
            'train': converted_files[:train_end],
            'val': converted_files[train_end:val_end],
            'test': converted_files[val_end:],
        }
        
        print(f"\n划分结果:")
        print(f"  训练集: {len(splits['train'])} ({self.train_ratio*100:.0f}%)")
        print(f"  验证集: {len(splits['val'])} ({self.val_ratio*100:.0f}%)")
        print(f"  测试集: {len(splits['test'])} ({self.test_ratio*100:.0f}%)")
        
        return splits
    
    def create_dataset_structure(self, splits: Dict[str, List]):
        """
        创建标准YOLO数据集目录结构
        
        Args:
            splits: 划分结果
        """
        print("\n" + "=" * 60)
        print("创建数据集目录结构")
        print("=" * 60)
        
        # 创建目录
        dirs = {}
        for split_name in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split_name
            label_dir = self.output_dir / 'labels' / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            dirs[split_name] = (img_dir, label_dir)
            print(f"  创建: images/{split_name}/, labels/{split_name}/")
        
        # 复制文件
        print("\n复制文件...")
        
        for split_name, (img_dir, label_dir) in dirs.items():
            for stem, label_path in splits[split_name]:
                # 复制图片
                src_img = self.img_dir / f'{stem}.tif'
                dst_img = img_dir / f'{stem}.jpg'  # 转换为JPEG
                
                if src_img.exists():
                    # 转换为JPEG
                    img = Image.open(src_img)
                    img.convert('RGB').save(dst_img, 'JPEG')
                
                # 复制标签
                dst_label = label_dir / f'{stem}.txt'
                if label_path.exists():
                    shutil.copy(label_path, dst_label)
            
            print(f"  {split_name}: {len(splits[split_name])} 个样本")
        
        print("\n数据集结构创建完成!")
        
        return dirs
    
    def generate_config(self) -> str:
        """生成YOLO数据集配置文件"""
        print("\n" + "=" * 60)
        print("生成数据集配置")
        print("=" * 60)
        
        config_content = f'''# ============================================================
#  泥石流/滑坡地质灾害识别 - 数据集配置文件
#  GeoHazardDet Dataset Configuration
# ============================================================
#
# 【说明】
# - 本配置文件用于YOLOv8模型训练
# - 适配2类地质灾害目标检测任务
# - 类别ID: 0=泥石流(debris_flow), 1=滑坡(landslide)
#
# 【使用方式】
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# model.train(data='configs/disaster.yaml', epochs=100)
#
# ============================================================

# ============ 数据集根目录 ============
# 使用相对路径，相对于项目根目录
path: ./datasets

# ============ 训练/验证/测试集路径 ============
train: images/train    # 训练集图片目录
val: images/val        # 验证集图片目录
test: images/test      # 测试集图片目录（可选）

# ============ 类别名称 ============
# 共2类地质灾害目标
names:
  # 类别ID: 0 - 泥石流
  0: debris_flow
  
  # 类别ID: 1 - 滑坡
  1: landslide

# ============ 类别数量 ============
nc: 2

# ============ 数据格式说明 ============
# YOLO格式标签文件 (.txt) 结构：
# <class_id> <x_center> <y_center> <width> <height>
# - class_id: 整数，0或1
# - x_center, y_center: 归一化中心坐标 (0-1)
# - width, height: 归一化宽高 (0-1)
#
# 示例（检测到1个泥石流目标）:
# 0 0.5 0.5 0.3 0.4
'''
        
        config_path = Path('configs/disaster.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"  配置已保存: {config_path}")
        
        return config_content
    
    def prepare(self, convert_limit: int = None):
        """
        完整的数据集准备流程
        
        Args:
            convert_limit: 限制转换数量
        """
        print("\n" + "=" * 60)
        print("GeoHazardDet 数据集准备")
        print("=" * 60)
        
        # 1. 分析原始数据
        analysis = self.analyze_original_data()
        
        # 2. 转换为YOLO格式
        converted_files, convert_stats = self.convert_mask_to_yolo(limit=convert_limit)
        
        if not converted_files:
            print("错误: 转换失败，无有效文件")
            return None
        
        # 3. 划分数据集
        splits = self.split_dataset(converted_files)
        
        # 4. 创建目录结构
        self.create_dataset_structure(splits)
        
        # 5. 生成配置
        self.generate_config()
        
        # 6. 统计信息
        print("\n" + "=" * 60)
        print("数据集准备完成!")
        print("=" * 60)
        print(f"\n最终数据集:")
        print(f"  总样本数: {len(converted_files)}")
        print(f"  训练集: {len(splits['train'])} ({self.train_ratio*100:.0f}%)")
        print(f"  验证集: {len(splits['val'])} ({self.val_ratio*100:.0f}%)")
        print(f"  测试集: {len(splits['test'])} ({self.test_ratio*100:.0f}%)")
        print(f"\n目录结构:")
        print(f"  datasets/")
        print(f"  ├── images/")
        print(f"  │   ├── train/ ({len(splits['train'])}张)")
        print(f"  │   ├── val/ ({len(splits['val'])}张)")
        print(f"  │   └── test/ ({len(splits['test'])}张)")
        print(f"  └── labels/")
        print(f"      ├── train/ ({len(splits['train'])}个)")
        print(f"      ├── val/ ({len(splits['val'])}个)")
        print(f"      └── test/ ({len(splits['test'])}个)")
        print(f"\n配置文件: configs/disaster.yaml")
        print(f"\n开始训练:")
        print(f"  python src/train.py --data configs/disaster.yaml --epochs 100")
        
        return {
            'analysis': analysis,
            'convert_stats': convert_stats,
            'splits': splits,
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GeoHazardDet数据集准备')
    parser.add_argument('--img-dir', default='datasets/img/', help='图片目录')
    parser.add_argument('--label-dir', default='datasets/label/', help='标签目录')
    parser.add_argument('--output-dir', default='datasets/', help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--limit', type=int, default=None, help='限制转换数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    result = preparer.prepare(convert_limit=args.limit)
    
    return result


if __name__ == '__main__':
    main()
