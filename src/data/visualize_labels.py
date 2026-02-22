# -*- coding: utf-8 -*-
"""
============================================================
数据集测试与标注可视化脚本
测试数据集完整性并生成标注可视化图片
============================================================

【功能】
- 检查数据集完整性
- 读取TIF图片和标签
- 生成带标注的可视化图片

【使用】
python src/data/visualize_labels.py

============================================================
"""

import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_label_binary(label_path: Path) -> List[Dict]:
    """
    读取二进制标签文件
    
    YOLO二进制格式: 每个目标5个float32 (class_id, x_center, y_center, width, height)
    
    Args:
        label_path: 标签文件路径
        
    Returns:
        标签列表 [{class_id, x_center, y_center, width, height}, ...]
    """
    if not label_path.exists():
        print(f"  [警告] 标签文件不存在: {label_path}")
        return []
    
    labels = []
    
    try:
        with open(label_path, 'rb') as f:
            data = f.read()
        
        # 检查数据长度
        if len(data) < 20:
            print(f"  [警告] 文件太小，不是有效的标签文件")
            return []
        
        # 解析float32数据 (每5个float32为一个标注)
        num_floats = len(data) // 4
        
        # 使用struct解析
        fmt = f'{num_floats}f'
        floats = struct.unpack(fmt, data)
        
        # 每5个为一组
        num_boxes = num_floats // 5
        
        print(f"  [信息] 二进制格式: {num_boxes} 个目标")
        
        for i in range(num_boxes):
            idx = i * 5
            labels.append({
                'class_id': int(floats[idx]),
                'x_center': float(floats[idx + 1]),
                'y_center': float(floats[idx + 2]),
                'width': float(floats[idx + 3]),
                'height': float(floats[idx + 4]),
            })
        
        return labels
        
    except Exception as e:
        print(f"  [错误] 读取失败: {e}")
        return []


def read_tiff_image(img_path: Path) -> Optional[np.ndarray]:
    """
    读取TIF图片
    
    Args:
        img_path: 图片路径
        
    Returns:
        numpy数组 or None
    """
    if not img_path.exists():
        print(f"  [警告] 图片文件不存在: {img_path}")
        return None
    
    try:
        # 使用PIL/Pillow读取
        from PIL import Image
        
        print(f"  [信息] 使用PIL读取: {img_path.name}")
        img = Image.open(str(img_path))
        
        # 获取基本信息
        width, height = img.size
        mode = img.mode
        print(f"  [信息] 原始尺寸: {width}x{height}, 模式: {mode}")
        
        # 转换为RGB数组
        if mode == 'I;16':
            # 16位灰度图转换
            img_array = np.array(img, dtype=np.uint16)
            # 归一化到0-255
            if img_array.max() > 0:
                img_array = (img_array / img_array.max() * 255).astype(np.uint8)
            img_array = np.stack([img_array] * 3, axis=2)
        elif mode == 'L':
            # 8位灰度图
            img_array = np.array(img.convert('RGB'))
        elif mode == 'RGB':
            img_array = np.array(img)
        elif mode == 'RGBA':
            img_array = np.array(img.convert('RGB'))
        else:
            # 其他模式尝试转换
            img_array = np.array(img.convert('RGB'))
        
        return img_array
        
    except ImportError as e:
        print(f"  [错误] PIL未安装: {e}")
        return None
    except Exception as e:
        print(f"  [错误] 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_labels(
    img: np.ndarray,
    labels: List[Dict],
    output_path: Path,
    class_names: Dict[int, str] = {0: '泥石流', 1: '滑坡'},
) -> bool:
    """
    可视化标注并保存图片
    
    Args:
        img: 图片数组
        labels: 标签列表
        output_path: 输出路径
        class_names: 类别名称
        
    Returns:
        是否成功
    """
    try:
        from PIL import Image, ImageDraw
        
        # 确保图片是uint8类型
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # 处理灰度图
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=2)
        elif img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=2)
        
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        img_width, img_height = pil_img.size
        
        # 颜色配置
        colors = {
            0: (255, 0, 0),      # 红色 - 泥石流
            1: (0, 255, 0),      # 绿色 - 滑坡
        }
        
        # 类别名称映射
        class_display_names = {
            0: 'Mudslide',
            1: 'Landslide',
        }
        
        print(f"  [信息] 绘制 {len(labels)} 个目标:")
        
        # 绘制每个标注
        for i, label in enumerate(labels):
            cls_id = label['class_id']
            xc, yc = label['x_center'], label['y_center']
            w, h = label['width'], label['height']
            
            # YOLO格式转像素坐标
            x1 = int((xc - w / 2) * img_width)
            y1 = int((yc - h / 2) * img_height)
            x2 = int((xc + w / 2) * img_width)
            y2 = int((yc + h / 2) * img_height)
            
            # 确保坐标在范围内
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)
            
            color = colors.get(cls_id, (255, 255, 0))
            class_name = class_display_names.get(cls_id, f'Class_{cls_id}')
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 绘制标签
            label_text = f"{class_name}"
            try:
                text_bbox = draw.textbbox((0, 0), label_text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = 80, 20
            
            # 标签背景
            draw.rectangle(
                [x1, y1 - text_height - 5, x1 + text_width + 10, y1],
                fill=color
            )
            
            # 标签文字
            draw.text(
                (x1 + 5, y1 - text_height - 3),
                label_text,
                fill=(255, 255, 255)
            )
            
            print(f"    [{i+1}] {class_name}: 坐标({x1},{y1},{x2},{y2})")
        
        # 保存图片
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)
        print(f"  [成功] 保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"  [错误] 可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_dataset(
    img_dir: str = 'datasets/img/',
    label_dir: str = 'datasets/label/',
    sample_count: int = 5,
) -> Dict:
    """
    分析数据集完整性
    """
    print("=" * 60)
    print("数据集完整性分析")
    print("=" * 60)
    
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    
    # 统计文件
    img_files = {}
    for f in img_dir.glob('*'):
        if f.suffix.lower() in ['.tif', '.tiff']:
            img_files[f.stem] = f
    
    label_files = {}
    for f in label_dir.glob('*'):
        if f.suffix.lower() in ['.tif', '.tiff']:
            label_files[f.stem] = f
    
    print(f"\n[文件统计]")
    print(f"  - 图片文件: {len(img_files)} 个")
    print(f"  - 标签文件: {len(label_files)} 个")
    
    # 配对情况
    matched = set(img_files.keys()) & set(label_files.keys())
    print(f"\n[配对情况]")
    print(f"  - 图片-标签配对: {len(matched)} 对")
    
    unmatched_img = set(img_files.keys()) - set(label_files.keys())
    if unmatched_img:
        print(f"  - 无标签的图片: {len(unmatched_img)} 个")
    
    # 标签统计
    total_objects = 0
    class_counts = {0: 0, 1: 0}
    
    print(f"\n[标签分析] (采样 {sample_count} 个文件)")
    
    sample_files = list(matched)[:sample_count]
    
    for i, stem in enumerate(sample_files):
        label_path = label_files[stem]
        print(f"\n  [{i+1}/{len(sample_files)}] {stem}")
        
        labels = read_label_binary(label_path)
        total_objects += len(labels)
        
        for label in labels:
            cls_id = label['class_id']
            if cls_id in class_counts:
                class_counts[cls_id] += 1
    
    print(f"\n[类别统计] (采样)")
    for cls_id, count in class_counts.items():
        name = '泥石流' if cls_id == 0 else '滑坡'
        print(f"  - {name}({cls_id}): {count} 个")
    
    return {
        'total_images': len(img_files),
        'total_labels': len(label_files),
        'matched_pairs': len(matched),
        'total_objects': total_objects,
        'class_counts': class_counts,
    }


def create_visualization(
    img_name: str,
    img_dir: str = 'datasets/img/',
    label_dir: str = 'datasets/label/',
    output_dir: str = 'runs/visualize/',
) -> bool:
    """
    创建单个图片的标注可视化
    """
    print("\n" + "=" * 60)
    print("生成标注可视化")
    print("=" * 60)
    
    img_path = Path(img_dir) / f"{img_name}.tif"
    label_path = Path(label_dir) / f"{img_name}.tif"
    output_path = Path(output_dir) / f"{img_name}_labeled.jpg"
    
    print(f"\n[处理] {img_name}")
    
    # 读取图片
    print("  [1/3] 读取图片...")
    img = read_tiff_image(img_path)
    
    if img is None:
        print("  [错误] 无法读取图片")
        return False
    
    print(f"  [信息] 图片尺寸: {img.shape}")
    
    # 读取标签
    print("  [2/3] 读取标签...")
    labels = read_label_binary(label_path)
    
    if not labels:
        print("  [警告] 无有效标签")
        # 保存原图
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img, mode='L')
            else:
                pil_img = Image.fromarray(img)
            pil_img.save(output_path)
            print(f"  [保存] 原图已保存: {output_path}")
        except:
            pass
        return True
    
    # 可视化
    print("  [3/3] 生成可视化...")
    return visualize_labels(img, labels, output_path)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集测试与标注可视化')
    parser.add_argument('--img-dir', default='datasets/img/', help='图片目录')
    parser.add_argument('--label-dir', default='datasets/label/', help='标签目录')
    parser.add_argument('--sample', type=int, default=5, help='采样数量')
    parser.add_argument('--img-name', type=str, default='moxizheng_0.2m_UAV0069', 
                       help='指定图片名称')
    
    args = parser.parse_args()
    
    # 分析数据集
    stats = analyze_dataset(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        sample_count=args.sample,
    )
    
    # 生成可视化
    success = create_visualization(
        img_name=args.img_name,
        img_dir=args.img_dir,
        label_dir=args.label_dir,
    )
    
    if success:
        print("\n" + "=" * 60)
        print("[完成] 可视化完成!")
        print(f"结果: runs/visualize/{args.img_name}_labeled.jpg")
    else:
        print("\n[失败] 可视化失败")


if __name__ == '__main__':
    main()
