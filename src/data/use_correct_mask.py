# -*- coding: utf-8 -*-
"""
============================================================
使用正确的mask目录重新生成标注
============================================================
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

def analyze_mask_directory():
    """分析mask目录"""
    print("=" * 60)
    print("Mask目录深度分析")
    print("=" * 60)
    
    mask_dir = Path('datasets/mask')
    img_dir = Path('datasets/img')
    
    # 获取mask文件
    mask_files = sorted(list(mask_dir.glob('*.tif')))
    img_files = sorted(list(img_dir.glob('*.tif')))
    
    print(f"\nMask目录: {len(mask_files)} 文件")
    print(f"Img目录: {len(img_files)} 文件")
    
    # 检查配对
    mask_stems = set(f.stem for f in mask_files)
    img_stems = set(f.stem for f in img_files)
    matched = mask_stems & img_stems
    
    print(f"配对文件: {len(matched)} 对")
    
    # 分析mask的唯一值分布
    print("\nMask值分布分析:")
    value_stats = defaultdict(int)
    
    for i, mask_file in enumerate(mask_files[:20]):  # 分析前20个
        arr = np.array(Image.open(mask_file))
        unique_vals = np.unique(arr)
        
        for v in unique_vals:
            value_stats[v] += 1
    
    print(f"唯一值: {sorted(value_stats.keys())}")
    for v, count in sorted(value_stats.items()):
        print(f"  值={v}: 出现在 {count} 个文件中")
    
    return mask_files, matched


def analyze_mask_values():
    """分析mask中每个文件的详细值"""
    print("\n" + "=" * 60)
    print("每个Mask文件的详细分析")
    print("=" * 60)
    
    mask_dir = Path('datasets/mask')
    samples = [
        'moxizheng_0.2m_UAV1690',
        'moxizheng_0.2m_UAV1680',
        'moxizheng_0.2m_UAV0069',
    ]
    
    for sample in samples:
        mask_path = mask_dir / f'{sample}.tif'
        
        if not mask_path.exists():
            print(f"[{sample}] 文件不存在")
            continue
            
        arr = np.array(Image.open(mask_path))
        
        print(f"\n[样本] {sample}")
        print(f"  形状: {arr.shape}")
        print(f"  数据类型: {arr.dtype}")
        print(f"  值范围: {arr.min()} - {arr.max()}")
        
        unique_vals = np.unique(arr)
        print(f"  唯一值: {unique_vals}")
        
        for v in unique_vals:
            count = (arr == v).sum()
            pct = count / arr.size * 100
            print(f"    值={v}: {count} 像素 ({pct:.2f}%)")


def check_mask_vs_image():
    """检查mask和原图的位置对应关系"""
    print("\n" + "=" * 60)
    print("Mask与原图位置对应分析")
    print("=" * 60)
    
    mask_dir = Path('datasets/mask')
    img_dir = Path('datasets/img')
    
    sample = 'moxizheng_0.2m_UAV1680'
    
    mask_path = mask_dir / f'{sample}.tif'
    img_path = img_dir / f'{sample}.tif'
    
    mask_arr = np.array(Image.open(mask_path))
    img_arr = np.array(Image.open(img_path))
    
    print(f"\n[样本] {sample}")
    print(f"Mask形状: {mask_arr.shape}")
    print(f"Img形状: {img_arr.shape}")
    
    # 找到mask中值为1的区域
    target_pixels = np.where(mask_arr == 1)
    
    print(f"\n目标区域像素数: {len(target_pixels[0])}")
    
    if len(target_pixels[0]) > 0:
        # 计算目标区域的边界
        y_min, y_max = target_pixels[0].min(), target_pixels[0].max()
        x_min, x_max = target_pixels[1].min(), target_pixels[1].max()
        
        print(f"目标区域边界: Y[{y_min}-{y_max}], X[{x_min}-{x_max}]")
        print(f"目标区域尺寸: {y_max - y_min + 1} x {x_max - x_min + 1}")
        
        # 检查原图中对应位置的像素值
        target_region = img_arr[y_min:y_max+1, x_min:x_max+1]
        print(f"对应原图区域形状: {target_region.shape}")
        print(f"原图区域值范围: {target_region.min()} - {target_region.max()}")


def visualize_with_correct_mask():
    """使用正确的mask生成可视化"""
    print("\n" + "=" * 60)
    print("使用正确的Mask生成可视化")
    print("=" * 60)
    
    from PIL import ImageDraw
    
    sample = 'moxizheng_0.2m_UAV0069'
    
    img_path = Path(f'datasets/img/{sample}.tif')
    mask_path = Path(f'datasets/mask/{sample}.tif')
    output_dir = Path('runs/visualize')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[处理] {sample}")
    
    # 读取原图
    img = Image.open(img_path)
    img_array = np.array(img)
    print(f"  原图: {img.size}")
    
    # 读取正确的mask
    mask = Image.open(mask_path)
    mask_array = np.array(mask)
    print(f"  Mask: {mask.size}, 唯一值: {np.unique(mask_array)}")
    
    # 创建彩色mask
    h, w = mask_array.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # mask中1的位置标为红色
    color_mask[mask_array == 1] = [255, 0, 0]  # 红色
    color_mask[mask_array == 0] = [0, 0, 0]   # 黑色
    
    # 合并原图和mask
    if len(img_array.shape) == 2:
        img_rgb = np.stack([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:
        img_rgb = img_array[:, :, :3]
    else:
        img_rgb = img_array
    
    # 归一化
    if img_rgb.max() > 255:
        img_rgb = (img_rgb / img_rgb.max() * 255).astype(np.uint8)
    
    # 叠加
    alpha = 0.5
    overlaid = (img_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    
    # 创建带标注的图
    annotated = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(annotated)
    
    # 找到所有目标区域
    for val in [1]:  # 只检查值为1的区域
        ys, xs = np.where(mask_array == val)
        
        if len(xs) == 0:
            continue
        
        # 计算边界框
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        
        # 标签
        label = "Target Area"
        draw.rectangle([x1, y1-25, x1+100, y1], fill=(255, 0, 0))
        draw.text((x1+5, y1-20), label, fill=(255, 255, 255))
        
        print(f"  目标区域: X[{x1}-{x2}], Y[{y1}-{y2}]")
    
    # 保存结果
    Image.fromarray(color_mask).save(output_dir / f'{sample}_correct_mask.jpg')
    Image.fromarray(overlaid).save(output_dir / f'{sample}_correct_overlay.jpg')
    annotated.save(output_dir / f'{sample}_correct_annotated.jpg')
    
    print(f"  已保存:")
    print(f"    - {sample}_correct_mask.jpg")
    print(f"    - {sample}_correct_overlay.jpg")
    print(f"    - {sample}_correct_annotated.jpg")
    
    return True


def generate_yolo_labels_from_mask():
    """从正确的mask生成YOLO标签"""
    print("\n" + "=" * 60)
    print("从Mask生成YOLO标签")
    print("=" * 60)
    
    mask_dir = Path('datasets/mask')
    img_dir = Path('datasets/img')
    output_dir = Path('datasets/labels_from_mask')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取配对文件
    mask_files = list(mask_dir.glob('*.tif'))
    
    print(f"\n处理 {len(mask_files)} 个文件...")
    
    stats = {'success': 0, 'no_target': 0}
    
    for mask_file in mask_files:
        stem = mask_file.stem
        
        # 读取mask
        mask_arr = np.array(Image.open(mask_file))
        
        # 检查唯一值
        unique_vals = np.unique(mask_arr)
        
        boxes = []
        
        # 处理值为1的区域
        if 1 in unique_vals:
            ys, xs = np.where(mask_arr == 1)
            
            if len(xs) > 0:
                h, w = mask_arr.shape
                
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                box_width = (x2 - x1) / w
                box_height = (y2 - y1) / h
                
                # 假设1=滑坡或泥石流
                boxes.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
                stats['success'] += 1
        
        # 保存YOLO标签
        output_path = output_dir / f'{stem}.txt'
        
        if boxes:
            with open(output_path, 'w') as f:
                f.write('\n'.join(boxes))
        else:
            stats['no_target'] += 1
    
    print(f"\n完成!")
    print(f"  成功转换: {stats['success']}")
    print(f"  无目标: {stats['no_target']}")
    print(f"  输出目录: {output_dir}")
    
    return stats


def main():
    """主函数"""
    # 分析mask目录
    analyze_mask_directory()
    
    # 分析mask值
    analyze_mask_values()
    
    # 检查位置对应
    check_mask_vs_image()
    
    # 使用正确的mask生成可视化
    visualize_with_correct_mask()
    
    # 生成YOLO标签
    stats = generate_yolo_labels_from_mask()
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    
    return stats


if __name__ == '__main__':
    main()
