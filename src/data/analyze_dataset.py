"""
============================================================
数据集测试与标注可视化
分析分割掩码格式并生成可视化.
============================================================
"""

from pathlib import Path

import numpy as np
from PIL import Image


def analyze_dataset():
    """分析数据集."""
    print("=" * 60)
    print("数据集分析")
    print("=" * 60)

    img_dir = Path("datasets/img")
    label_dir = Path("datasets/label")

    # 统计文件
    img_files = list(img_dir.glob("*.tif"))
    label_files = list(label_dir.glob("*.tif"))

    print("\n[文件统计]")
    print(f"  图片文件: {len(img_files)} 个")
    print(f"  标签文件: {len(label_files)} 个")

    # 检查配对
    img_stems = set(f.stem for f in img_files)
    label_stems = set(f.stem for f in label_files)

    matched = img_stems & label_stems

    print("\n[配对情况]")
    print(f"  图片-标签配对: {len(matched)} 对")

    return list(img_files), list(label_files)


def visualize_mask_sample():
    """可视化分割掩码样本."""
    print("\n" + "=" * 60)
    print("生成掩码可视化")
    print("=" * 60)

    # 选择一个样本
    sample_name = "moxizheng_0.2m_UAV0069"

    img_path = Path(f"datasets/img/{sample_name}.tif")
    label_path = Path(f"datasets/label/{sample_name}.tif")
    output_dir = Path("runs/visualize")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[处理] {sample_name}")

    # 1. 读取原图
    print("  [1/3] 读取原图...")
    img = Image.open(img_path)
    img_array = np.array(img)
    print(f"       尺寸: {img.size}, 模式: {img.mode}, 数组形状: {img_array.shape}")

    # 2. 读取标签/掩码
    print("  [2/3] 读取掩码...")
    mask = Image.open(label_path)
    mask_array = np.array(mask)
    print(f"       尺寸: {mask.size}, 模式: {mask.mode}, 数组形状: {mask_array.shape}")

    # 分析掩码 - 将RGB转为灰度
    if len(mask_array.shape) == 3:
        mask_gray = mask_array[:, :, 0]  # 取第一个通道
    else:
        mask_gray = mask_array

    unique_values = np.unique(mask_gray)
    print(f"       唯一值: {unique_values}")
    for v in unique_values:
        count = (mask_gray == v).sum()
        print(f"         值={v}: {count} 像素 ({count / mask_gray.size * 100:.2f}%)")

    # 3. 生成可视化
    print("  [3/3] 生成可视化...")

    # 创建彩色掩码
    h, w = mask_gray.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 根据像素值着色
    for val in unique_values:
        if val == 0:
            continue  # 背景保持黑色
        elif val == 255:
            color_mask[mask_gray == val] = [255, 0, 0]  # 红色 - 泥石流

    # 合并原图
    if len(img_array.shape) == 2:
        img_rgb = np.stack([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:
        img_rgb = img_array[:, :, :3]
    else:
        img_rgb = img_array

    # 归一化到0-255
    if img_rgb.max() > 255:
        img_rgb = (img_rgb / img_rgb.max() * 255).astype(np.uint8)

    # 混合
    alpha = 0.5
    overlaid = (img_rgb * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    # 找到目标区域
    targets = []
    for val in unique_values:
        if val == 0:
            continue
        ys, xs = np.where(mask_gray == val)
        if len(xs) > 0:
            center_x = xs.mean()
            center_y = ys.mean()
            class_name = "Mudslide" if val == 255 else f"Class_{val}"
            targets.append(
                {
                    "class_id": 0,  # 统一为类别0
                    "name": class_name,
                    "center": (center_x, center_y),
                    "count": len(xs),
                }
            )
            print(f"       {class_name}: 中心({center_x:.0f}, {center_y:.0f}), {len(xs)}像素")

    # 保存结果
    # 1. 原图
    Image.fromarray(img_rgb).save(output_dir / f"{sample_name}_original.jpg")
    print(f"       已保存: {sample_name}_original.jpg")

    # 2. 掩码图
    Image.fromarray(color_mask).save(output_dir / f"{sample_name}_mask.jpg")
    print(f"       已保存: {sample_name}_mask.jpg")

    # 3. 叠加图
    Image.fromarray(overlaid).save(output_dir / f"{sample_name}_overlaid.jpg")
    print(f"       已保存: {sample_name}_overlaid.jpg")

    # 4. 带标注的图
    from PIL import ImageDraw

    annotated = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(annotated)

    for i, target in enumerate(targets):
        color = (255, 0, 0)

        # 绘制边界框
        ys, xs = np.where(mask_gray == 255)
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 绘制标签
        label = "Mudslide"
        draw.rectangle([x1, y1 - 25, x1 + 90, y1], fill=color)
        draw.text((x1 + 5, y1 - 20), label, fill=(255, 255, 255))

        print(f"       标注{i + 1}: Mudslide, 边界框({x1},{y1},{x2},{y2})")

    annotated.save(output_dir / f"{sample_name}_annotated.jpg")
    print(f"       已保存: {sample_name}_annotated.jpg")

    return True


def convert_mask_to_yolo():
    """将分割掩码转换为YOLO边界框格式."""
    print("\n" + "=" * 60)
    print("转换分割掩码为YOLO边界框")
    print("=" * 60)

    label_dir = Path("datasets/label")
    output_dir = Path("datasets/labels_yolo")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(label_dir.glob("*.tif"))

    print(f"\n转换 {len(label_files)} 个文件...")

    converted_count = 0
    for label_path in label_files[:10]:  # 先转换10个样本
        stem = label_path.stem

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

            # 类别ID: 假设255=泥石流(0), 其他=滑坡(1)
            class_id = 0 if val == 255 else 1

            boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 保存YOLO格式标签
        if boxes:
            output_path = output_dir / f"{stem}.txt"
            with open(output_path, "w") as f:
                f.write("\n".join(boxes))

            converted_count += 1
            print(f"  {stem}: {len(boxes)} 个目标")

    print(f"\n完成! 转换了 {converted_count} 个文件到 {output_dir}")

    return converted_count


def main():
    """主函数."""
    # 1. 分析数据集
    _img_files, _label_files = analyze_dataset()

    # 2. 可视化样本
    visualize_mask_sample()

    # 3. 转换格式
    convert_mask_to_yolo()

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print("\n可视化结果保存在: runs/visualize/")
    print("YOLO标签保存在: datasets/labels_yolo/")


if __name__ == "__main__":
    main()
