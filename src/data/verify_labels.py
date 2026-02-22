# -*- coding: utf-8 -*-
"""
============================================================
验证生成的YOLO标签是否正确
============================================================
"""

from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path


def visualize_samples():
    """可视化多个样本以验证标签正确性"""
    
    # 选择几个样本进行验证
    samples = [
        ('datasets/images/val/moxizheng_0.2m_UAV1680.jpg', 
         'datasets/labels/val/moxizheng_0.2m_UAV1680.txt',
         '验证集样本1 (UAV1680)'),
        
        ('datasets/images/train/moxizheng_0.2m_UAV0001.jpg',
         'datasets/labels/train/moxizheng_0.2m_UAV0001.txt',
         '训练集样本1 (UAV0001)'),
        
        ('datasets/images/test/moxizheng_0.2m_UAV0069.jpg',
         'datasets/labels/test/moxizheng_0.2m_UAV0069.txt',
         '测试集样本1 (UAV0069)'),
    ]
    
    output_dir = Path('runs/verify_labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path_str, label_path_str, title in samples:
        img_path = Path(img_path_str)
        label_path = Path(label_path_str)
        
        if not img_path.exists() or not label_path.exists():
            print(f"跳过: {img_path.name} (文件不存在)")
            continue
        
        print(f"\n处理: {title}")
        
        # 读取图片
        img = Image.open(img_path)
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=2)
        else:
            img_rgb = img_array.copy()
        
        if img_rgb.max() > 255:
            img_rgb = (img_rgb / img_rgb.max() * 255).astype(np.uint8)
        
        # 读取YOLO标签
        with open(label_path, 'r') as f:
            content = f.read().strip()
        
        print(f"  YOLO标签: {content}")
        
        # 绘制边界框
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        img_w, img_h = img.size
        
        for line in content.split('\n'):
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_w
            y_center = float(parts[2]) * img_h
            width = float(parts[3]) * img_w
            height = float(parts[4]) * img_h
            
            # 计算边界框坐标
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # 类别名称
            class_name = "泥石流/滑坡" if class_id == 0 else "类别1"
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 添加标签
            label_text = f"{class_name}: {class_id}"
            draw.rectangle([x1, y1-25, x1+120, y1], fill=color)
            draw.text((x1+5, y1-20), label_text, fill=(0, 0, 0))
            
            # 添加坐标信息
            info_text = f"({x1},{y1})-({x2},{y2})"
            draw.rectangle([x1, y2, x1+150, y2+20], fill=(0, 0, 0))
            draw.text((x1+5, y2+2), info_text, fill=(255, 255, 255))
            
            print(f"  边界框: X[{x1}-{x2}], Y[{y1}-{y2}]")
            print(f"  目标大小: {width:.0f} x {height:.0f} 像素")
            print(f"  目标比例: {width*height/(img_w*img_h)*100:.1f}%")
        
        # 保存图片
        safe_title = title.replace(' ', '_').replace('(', '').replace(')', '')
        output_name = f"{safe_title}.jpg"
        output_path = output_dir / output_name
        pil_img.save(output_path)
        
        print(f"  已保存: {output_path}")
    
    return output_dir


def check_all_labels():
    """检查所有标签的统计信息"""
    print("\n" + "=" * 60)
    print("标签统计信息")
    print("=" * 60)
    
    from collections import defaultdict
    
    all_stats = {
        'total_labels': 0,
        'class_counts': defaultdict(int),
        'sizes': [],
    }
    
    for split in ['train', 'val', 'test']:
        label_dir = Path(f'datasets/labels/{split}')
        
        if not label_dir.exists():
            continue
        
        split_count = 0
        
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                content = f.read().strip()
            
            for line in content.split('\n'):
                if not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                width = float(parts[3])
                height = float(parts[4])
                
                all_stats['total_labels'] += 1
                all_stats['class_counts'][class_id] += 1
                all_stats['sizes'].append(width * height * 100)
                
                split_count += 1
        
        print(f"  {split}: {split_count} 个标签")
    
    print(f"\n总计:")
    print(f"  总标签数: {all_stats['total_labels']}")
    
    for cls_id, count in all_stats['class_counts'].items():
        name = "泥石流/滑坡" if cls_id == 0 else f"类别{cls_id}"
        print(f"  {name}: {count} 个")
    
    if all_stats['sizes']:
        import statistics
        print(f"\n目标大小统计:")
        print(f"  最小: {min(all_stats['sizes']):.2f}%")
        print(f"  最大: {max(all_stats['sizes']):.2f}%")
        print(f"  平均: {statistics.mean(all_stats['sizes']):.2f}%")
    
    return all_stats


if __name__ == '__main__':
    print("=" * 60)
    print("验证生成的YOLO标签")
    print("=" * 60)
    
    # 可视化样本
    output_dir = visualize_samples()
    
    # 统计信息
    all_stats = check_all_labels()
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)
    print(f"\n验证图片保存位置: {output_dir}")
