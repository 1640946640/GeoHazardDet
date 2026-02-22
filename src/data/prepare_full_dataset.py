# -*- coding: utf-8 -*-
"""
============================================================
完整数据集处理脚本
将mask掩码转换为YOLO格式，并生成训练集/验证集/测试集
============================================================

【功能】
1. 从mask目录读取所有标注
2. 转换为YOLO边界框格式
3. 自动划分训练集(70%)/验证集(20%)/测试集(10%)
4. 生成配置文件

【使用】
python src/data/prepare_full_dataset.py

============================================================
"""

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from PIL import Image


class FullDatasetPreparer:
    """完整数据集准备器"""
    
    # 类别配置
    CLASS_NAMES = {
        0: 'debris_flow',   # 泥石流
        1: 'landslide',     # 滑坡
    }
    
    def __init__(
        self,
        img_dir: str = 'datasets/img/',
        mask_dir: str = 'datasets/mask/',
        output_dir: str = 'datasets/',
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """初始化"""
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
    
    def load_all_data(self) -> Tuple[List[Path], Dict]:
        """
        加载所有配对的数据
        
        Returns:
            (文件列表, 统计信息)
        """
        print("=" * 60)
        print("加载数据")
        print("=" * 60)
        
        # 获取所有mask文件
        mask_files = list(self.mask_dir.glob('*.tif'))
        
        # 获取所有原图文件
        img_files = {f.stem: f for f in self.img_dir.glob('*.tif')}
        
        print(f"\nMask文件: {len(mask_files)}")
        print(f"原图文件: {len(img_files)}")
        
        # 配对
        paired = []
        for mask_file in mask_files:
            stem = mask_file.stem
            if stem in img_files:
                paired.append({
                    'stem': stem,
                    'img_path': img_files[stem],
                    'mask_path': mask_file,
                })
        
        print(f"配对文件: {len(paired)}")
        
        # 统计信息
        stats = {
            'total_paired': len(paired),
        }
        
        return paired, stats
    
    def process_masks(self, paired_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        处理所有mask，转换为YOLO格式
        
        Args:
            paired_data: 配对数据列表
            
        Returns:
            (处理后的数据, 统计信息)
        """
        print("\n" + "=" * 60)
        print("处理Mask标注")
        print("=" * 60)
        
        processed = []
        stats = {
            'success': 0,
            'no_target': 0,
            'class_counts': defaultdict(int),
            'target_sizes': [],
        }
        
        # 进度条
        for item in tqdm(paired_data, desc="处理标注"):
            stem = item['stem']
            mask_path = item['mask_path']
            
            try:
                # 读取mask
                mask = np.array(Image.open(mask_path))
                
                # 获取唯一值
                unique_vals = np.unique(mask)
                
                boxes = []
                
                # 处理每个非零值
                for val in unique_vals:
                    if val == 0:
                        continue
                    
                    # 找到该值的像素位置
                    ys, xs = np.where(mask == val)
                    
                    if len(xs) == 0:
                        continue
                    
                    # 计算边界框
                    h, w = mask.shape
                    
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    
                    # 归一化坐标
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h
                    
                    # 类别ID (1=滑坡/泥石流)
                    class_id = 0  # 假设所有mask值都是同一类别
                    
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': box_width,
                        'height': box_height,
                    })
                    
                    stats['class_counts'][class_id] += 1
                    stats['target_sizes'].append({
                        'width': box_width * 100,
                        'height': box_height * 100,
                        'area': box_width * box_height * 10000,
                    })
                
                if boxes:
                    processed.append({
                        'stem': stem,
                        'img_path': item['img_path'],
                        'boxes': boxes,
                    })
                    stats['success'] += 1
                else:
                    stats['no_target'] += 1
                    
            except Exception as e:
                print(f"\n错误: {stem} - {e}")
                stats['no_target'] += 1
        
        print(f"\n处理完成:")
        print(f"  成功: {stats['success']}")
        print(f"  无目标: {stats['no_target']}")
        
        # 目标大小统计
        if stats['target_sizes']:
            areas = [s['area'] for s in stats['target_sizes']]
            print(f"\n目标大小统计:")
            print(f"  最小: {min(areas):.2f}%")
            print(f"  最大: {max(areas):.2f}%")
            print(f"  平均: {sum(areas)/len(areas):.2f}%")
        
        return processed, stats
    
    def split_dataset(self, processed_data: List[Dict]) -> Dict[str, List]:
        """
        划分数据集
        
        Args:
            processed_data: 处理后的数据
            
        Returns:
            划分结果
        """
        print("\n" + "=" * 60)
        print("划分数据集")
        print("=" * 60)
        
        # 打乱数据
        random.shuffle(processed_data)
        
        total = len(processed_data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        splits = {
            'train': processed_data[:train_end],
            'val': processed_data[train_end:val_end],
            'test': processed_data[val_end:],
        }
        
        print(f"\n划分结果:")
        print(f"  总样本: {total}")
        print(f"  训练集: {len(splits['train'])} ({self.train_ratio*100:.0f}%)")
        print(f"  验证集: {len(splits['val'])} ({self.val_ratio*100:.0f}%)")
        print(f"  测试集: {len(splits['test'])} ({self.test_ratio*100:.0f}%)")
        
        return splits
    
    def create_dataset_structure(self, splits: Dict[str, List]):
        """
        创建数据集目录结构并复制文件
        
        Args:
            splits: 划分结果
            
        Returns:
            创建的目录路径
        """
        print("\n" + "=" * 60)
        print("创建数据集目录结构")
        print("=" * 60)
        
        dirs = {}
        
        for split_name in ['train', 'val', 'test']:
            img_dir = self.output_dir / 'images' / split_name
            label_dir = self.output_dir / 'labels' / split_name
            
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            dirs[split_name] = {
                'img': img_dir,
                'label': label_dir,
            }
            
            print(f"创建: images/{split_name}/, labels/{split_name}/")
        
        return dirs
    
    def copy_and_save_labels(
        self,
        splits: Dict[str, List],
        dirs: Dict[str, Dict],
    ) -> Dict:
        """
        复制图片并保存YOLO标签
        
        Args:
            splits: 数据划分
            dirs: 目录路径
            
        Returns:
            统计信息
        """
        print("\n" + "=" * 60)
        print("保存文件")
        print("=" * 60)
        
        stats = {}
        
        for split_name, data_list in splits.items():
            img_dir = dirs[split_name]['img']
            label_dir = dirs[split_name]['label']
            
            print(f"\n处理 {split_name} 集 ({len(data_list)} 个文件)...")
            
            for item in tqdm(data_list, desc=split_name):
                stem = item['stem']
                
                # 复制图片（转换为JPEG）
                src_img = item['img_path']
                dst_img = img_dir / f'{stem}.jpg'
                
                try:
                    img = Image.open(src_img)
                    img_rgb = img.convert('RGB')
                    img_rgb.save(dst_img, 'JPEG', quality=95)
                except Exception as e:
                    print(f"\n图片保存错误: {stem} - {e}")
                
                # 保存YOLO标签
                dst_label = label_dir / f'{stem}.txt'
                
                with open(dst_label, 'w') as f:
                    for box in item['boxes']:
                        f.write(
                            f"{box['class_id']} "
                            f"{box['x_center']:.6f} "
                            f"{box['y_center']:.6f} "
                            f"{box['width']:.6f} "
                            f"{box['height']:.6f}\n"
                        )
            
            stats[split_name] = len(data_list)
        
        print(f"\n文件保存完成!")
        
        return stats
    
    def generate_yaml_config(self) -> Path:
        """
        生成YAML配置文件
        
        Returns:
            配置文件的路径
        """
        print("\n" + "=" * 60)
        print("生成配置文件")
        print("=" * 60)
        
        config_content = f'''# ============================================================
#  泥石流/滑坡地质灾害识别 - YOLOv8数据集配置文件
#  GeoHazardDet Dataset Configuration
# ============================================================
#
# 【说明】
# - 本配置文件用于YOLOv8模型训练
# - 适配2类地质灾害目标检测任务
# - 类别ID: 0=泥石流(debris_flow), 1=滑坡(landslide)
#
# 【数据来源】
# - 图片: datasets/img/*.tif
# - 标注: datasets/mask/*.tif (二值分割掩码)
#
# 【使用方式】
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# model.train(data='configs/disaster.yaml', epochs=100)
#
# ============================================================

# ============ 数据集根目录 ============
path: ./datasets

# ============ 训练/验证/测试集路径 ============
train: images/train    # 训练集图片目录
val: images/val        # 验证集图片目录
test: images/test      # 测试集图片目录（可选）

# ============ 类别数量 ============
nc: 2

# ============ 类别名称 ============
names:
  # 类别ID: 0 - 泥石流
  0: debris_flow
  
  # 类别ID: 1 - 滑坡
  1: landslide

# ============ 数据格式说明 ============
# YOLO格式标签文件 (.txt) 结构：
# <class_id> <x_center> <y_center> <width> <height>
#
# 示例（检测到1个泥石流目标）:
# 0 0.5 0.5 0.3 0.4
#
# 标签生成方式：
# - 从 datasets/mask/ 目录读取二值掩码
# - 值为1的区域即为灾害目标区域
# - 自动计算最小外接矩形作为边界框
'''
        
        config_path = Path('configs/disaster.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"配置已保存: {config_path}")
        
        return config_path
    
    def generate_dataset_info(self, stats: Dict) -> str:
        """
        生成数据集信息报告

        Args:
            stats: 统计信息

        Returns:
            报告内容
        """
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        total = sum(stats.values())
        
        report = f"""
# ============================================================
# GeoHazardDet 数据集信息
# ============================================================

## 数据集统计

| 数据集 | 样本数 | 比例 |
|--------|--------|------|
| 训练集 | {stats['train']} | {self.train_ratio*100:.0f}% |
| 验证集 | {stats['val']} | {self.val_ratio*100:.0f}% |
| 测试集 | {stats['test']} | {self.test_ratio*100:.0f}% |
| **总计** | **{total}** | 100% |

## 类别信息

| ID | 名称 | 说明 |
|----|------|------|
| 0 | debris_flow | 泥石流 |
| 1 | landslide | 滑坡 |

## 数据格式

- **图片**: JPEG格式, 512×512像素
- **标签**: YOLO格式TXT文件
- **标注来源**: 从mask二值掩码提取边界框

## 目录结构

```
datasets/
├── images/
│   ├── train/  ({stats['train']}张图片)
│   ├── val/    ({stats['val']}张图片)
│   └── test/   ({stats['test']}张图片)
└── labels/
    ├── train/  ({stats['train']}个标签)
    ├── val/    ({stats['val']}个标签)
    └── test/   ({stats['test']}个标签)
```

## 使用方法

### 训练模型

```bash
python src/train.py --data configs/disaster.yaml --epochs 100
```

### 自定义训练

```bash
python src/train.py --data configs/disaster.yaml --epochs 200 --batch 16 --model yolov8n.pt
```

## 生成日期

{current_time}
"""
    
    def prepare(self):
        """
        执行完整的数据集准备流程
        """
        print("\n" + "=" * 60)
        print("GeoHazardDet 完整数据集准备")
        print("=" * 60)
        
        # 1. 加载数据
        paired_data, load_stats = self.load_all_data()
        
        if not paired_data:
            print("错误: 没有找到配对的数据文件!")
            return None
        
        # 2. 处理mask标注
        processed_data, process_stats = self.process_masks(paired_data)
        
        if not processed_data:
            print("错误: 没有成功处理任何标注!")
            return None
        
        # 3. 划分数据集
        splits = self.split_dataset(processed_data)
        
        # 4. 创建目录结构
        dirs = self.create_dataset_structure(splits)
        
        # 5. 复制文件并保存标签
        file_stats = self.copy_and_save_labels(splits, dirs)
        
        # 6. 生成配置文件
        self.generate_yaml_config()
        
        # 7. 生成数据集信息
        info_report = self.generate_dataset_info(file_stats)
        info_path = self.output_dir / 'dataset_info.md'
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(info_report)
        print(f"\n数据集信息: {info_path}")
        
        # 最终统计
        print("\n" + "=" * 60)
        print("数据集准备完成!")
        print("=" * 60)
        print(f"\n最终数据集:")
        print(f"  总样本: {sum(file_stats.values())}")
        print(f"  训练集: {file_stats['train']} ({self.train_ratio*100:.0f}%)")
        print(f"  验证集: {file_stats['val']} ({self.val_ratio*100:.0f}%)")
        print(f"  测试集: {file_stats['test']} ({self.test_ratio*100:.0f}%)")
        print(f"\n目录结构:")
        print(f"  datasets/")
        print(f"  ├── images/")
        print(f"  │   ├── train/ ({file_stats['train']}张)")
        print(f"  │   ├── val/ ({file_stats['val']}张)")
        print(f"  │   └── test/ ({file_stats['test']}张)")
        print(f"  └── labels/")
        print(f"      ├── train/ ({file_stats['train']}个)")
        print(f"      ├── val/ ({file_stats['val']}个)")
        print(f"      └── test/ ({file_stats['test']}个)")
        print(f"\n配置文件: configs/disaster.yaml")
        print(f"\n开始训练:")
        print(f"  python src/train.py --data configs/disaster.yaml --epochs 100")
        
        return {
            'paired': load_stats['total_paired'],
            'processed': process_stats['success'],
            'splits': file_stats,
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GeoHazardDet完整数据集准备')
    parser.add_argument('--img-dir', default='datasets/img/', help='图片目录')
    parser.add_argument('--mask-dir', default='datasets/mask/', help='掩码目录')
    parser.add_argument('--output-dir', default='datasets/', help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    preparer = FullDatasetPreparer(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    result = preparer.prepare()
    
    return result


if __name__ == '__main__':
    main()
