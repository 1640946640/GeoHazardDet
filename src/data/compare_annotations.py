"""
============================================================
生成正确vs错误的标注对比图.
============================================================
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_comparison():
    """创建标注对比图."""
    sample = "moxizheng_0.2m_UAV0069"

    # 读取原图
    img_path = Path(f"datasets/img/{sample}.tif")
    img = Image.open(img_path)
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_rgb = np.stack([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:
        img_rgb = img_array[:, :, :3]
    else:
        img_rgb = img_array

    if img_rgb.max() > 255:
        img_rgb = (img_rgb / img_rgb.max() * 255).astype(np.uint8)

    # 转换为PIL Image
    pil_img = Image.fromarray(img_rgb)

    # 读取正确的mask
    mask_path = Path(f"datasets/mask/{sample}.tif")
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # 创建错误版本的副本（红色边框）
    img_wrong = pil_img.copy()
    draw_wrong = ImageDraw.Draw(img_wrong)

    # 错误：整个图片被标记（红色）
    draw_wrong.rectangle([0, 0, 511, 511], outline=(255, 0, 0), width=5)
    draw_wrong.rectangle([0, 0, 180, 25], fill=(255, 0, 0))
    draw_wrong.text((5, 5), "WRONG!", fill=(255, 255, 255))

    # 创建正确版本的副本（绿色边框）
    img_correct = pil_img.copy()
    draw_correct = ImageDraw.Draw(img_correct)

    # 正确：根据mask找到目标区域（绿色）
    ys, xs = np.where(mask_array == 1)

    if len(xs) > 0:
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()

        # 绘制正确的边界框
        draw_correct.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)

        # 标签
        draw_correct.rectangle([x1, y1 - 25, x1 + 150, y1], fill=(0, 255, 0))
        draw_correct.text((x1 + 5, y1 - 20), "CORRECT!", fill=(255, 255, 255))

    # 创建对比图
    # 左侧：错误版本，右侧：正确版本
    comparison = Image.new("RGB", (512 * 2 + 20, 512), (240, 240, 240))

    # 放置图片
    comparison.paste(img_wrong, (0, 0))
    comparison.paste(img_correct, (512 + 20, 0))

    # 添加标签文字
    draw_comp = ImageDraw.Draw(comparison)
    draw_comp.rectangle([0, 0, 512, 30], fill=(255, 200, 200))
    draw_comp.text((200, 8), "WRONG Annotation", fill=(255, 0, 0))

    draw_comp.rectangle([512 + 20, 0, 512 + 20 + 200, 30], fill=(200, 255, 200))
    draw_comp.text((512 + 20 + 60, 8), "CORRECT Annotation", fill=(0, 150, 0))

    # 保存对比图
    output_path = Path("runs/visualize/annotation_comparison.jpg")
    comparison.save(output_path)

    print(f"对比图已保存: {output_path}")

    # 保存单独的图片
    img_wrong.save("runs/visualize/wrong_annotation.jpg")
    img_correct.save("runs/visualize/correct_annotation.jpg")

    print("单独版本:")
    print("  - runs/visualize/wrong_annotation.jpg")
    print("  - runs/visualize/correct_annotation.jpg")

    return True


def show_label_difference():
    """显示标签文件的差异."""
    print("\n" + "=" * 60)
    print("标签目录对比")
    print("=" * 60)

    sample = "moxizheng_0.2m_UAV0069"

    # Label目录（旧/错误）
    print(f"\n[错误] datasets/label/{sample}.tif")
    print("  格式: RGB图片 (512x512x3)")
    print("  值: 只有0和255")
    print("  问题: 无法区分目标位置")

    # Mask目录（新/正确）
    print(f"\n[正确] datasets/mask/{sample}.tif")
    print("  格式: 二值掩码 (512x512)")
    print("  值: 0=背景, 1=目标")
    print("  优点: 精确标注目标位置")

    print("\n" + "=" * 60)
    print("正确的YOLO标签")
    print("=" * 60)

    label_path = Path(f"datasets/labels_from_mask/{sample}.tif.txt")
    if label_path.exists():
        with open(label_path) as f:
            content = f.read()
        print(f"文件: {label_path.name}")
        print(f"内容:\n{content}")
        print("\n解析:")
        parts = content.strip().split()
        print(f"  class_id: {parts[0]} (0=泥石流/滑坡)")
        print(f"  x_center: {parts[1]} ({float(parts[1]) * 100:.1f}% 图片宽度)")
        print(f"  y_center: {parts[2]} ({float(parts[2]) * 100:.1f}% 图片高度)")
        print(f"  width: {parts[3]} ({float(parts[3]) * 100:.1f}% 图片宽度)")
        print(f"  height: {parts[4]} ({float(parts[4]) * 100:.1f}% 图片高度)")

    return True


if __name__ == "__main__":
    create_comparison()
    show_label_difference()

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print("\n查看对比图: runs/visualize/annotation_comparison.jpg")
