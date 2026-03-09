# 详细分析标签文件，找出所有唯一值和区域
from collections import defaultdict

import numpy as np
from PIL import Image

# 选择多个样本进行分析
samples = [
    "moxizheng_0.2m_UAV0069",
    "moxizheng_0.2m_UAV1680",
    "moxizheng_0.2m_UAV0060",
    "moxizheng_0.2m_UAV0059",
]

print("=" * 60)
print("详细标签分析")
print("=" * 60)

for sample in samples:
    label_path = f"datasets/label/{sample}.tif"
    img = Image.open(label_path)
    arr = np.array(img)

    print(f"\n[样本] {sample}")
    print(f"  形状: {arr.shape}, 数据类型: {arr.dtype}")

    # 如果是RGB，找出所有唯一颜色
    if len(arr.shape) == 3:
        # 统计所有唯一RGB组合
        unique_colors = set()
        h, w, c = arr.shape
        for i in range(h):
            for j in range(w):
                pixel = tuple(arr[i, j])
                unique_colors.add(pixel)

        print(f"  RGB唯一颜色数: {len(unique_colors)}")

        # 统计每种颜色的像素数
        color_counts = defaultdict(int)
        for i in range(h):
            for j in range(w):
                pixel = tuple(arr[i, j])
                color_counts[pixel] += 1

        print("  颜色统计 (前10种):")
        for color, count in sorted(color_counts.items(), key=lambda x: -x[1])[:10]:
            pct = count / (h * w) * 100
            print(f"    {color}: {count} 像素 ({pct:.2f}%)")

    else:
        # 灰度图
        unique_vals = np.unique(arr)
        print(f"  唯一值: {unique_vals}")

        for v in unique_vals:
            count = (arr == v).sum()
            pct = count / arr.size * 100
            print(f"    值={v}: {count} 像素 ({pct:.2f}%)")

print("\n" + "=" * 60)
print("检查是否有多个独立目标区域")
print("=" * 60)

# 分析第一个样本的连通区域
from scipy import ndimage

sample = "moxizheng_0.2m_UAV0069"
label_path = f"datasets/label/{sample}.tif"
arr = np.array(Image.open(label_path))

# 转灰度
if len(arr.shape) == 3:
    gray = arr[:, :, 0]
else:
    gray = arr

# 创建二值图（非0像素）
binary = gray > 0

# 标记连通区域
labeled_array, num_features = ndimage.label(binary)

print(f"\n[连通区域分析] {sample}")
print(f"  非0像素总数: {(gray > 0).sum()}")
print(f"  独立目标区域数: {num_features}")

# 统计每个区域的大小
for region_id in range(1, min(num_features + 1, 10)):
    region_size = (labeled_array == region_id).sum()
    print(f"    区域{region_id}: {region_size} 像素")
