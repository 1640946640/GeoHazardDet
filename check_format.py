# 对比图片、标签、掩码文件大小
import os

from PIL import Image

img_path = "datasets/img/moxizheng_0.2m_UAV0069.tif"
label_path = "datasets/label/moxizheng_0.2m_UAV0069.tif"
mask_path = "datasets/mask/moxizheng_0.2m_UAV1690.tif"

img_size = os.path.getsize(img_path)
label_size = os.path.getsize(label_path)
mask_size = os.path.getsize(mask_path)

print(f"图片大小: {img_size:,} bytes ({img_size / 1024 / 1024:.2f} MB)")
print(f"标签大小: {label_size:,} bytes ({label_size / 1024 / 1024:.2f} MB)")
print(f"掩码大小: {mask_size:,} bytes ({mask_size / 1024 / 1024:.2f} MB)")

# 标签文件应该是5*4=20字节一个目标
print(f"\n如果按YOLO格式: {label_size / 20:.0f} 个目标")

# 检查图片尺寸
img = Image.open(img_path)
print(f"图片尺寸: {img.size[0]} x {img.size[1]}")

# 512*512 = 262144 像素
# 如果标签是512x512的分割掩码: 262144 * 4 = 1,048,576 bytes
print(f"图片大小对应: 512x512 RGB = {512 * 512 * 3:,} bytes")

# 检查label文件的实际内容
import struct

with open(label_path, "rb") as f:
    data = f.read()

print("\n标签文件分析:")
print(f"  总字节: {len(data)}")
print(f"  总float32: {len(data) // 4}")

# 检查是否有有效的class_id (应该是0或1)
floats = struct.unpack(f"{len(data) // 4}f", data)
valid_class_ids = sum(1 for f in floats[::5] if f == 0.0 or f == 1.0)
print(f"  有效的class_id (0或1): {valid_class_ids}")

# 检查前10个class_id
print(f"  前10个class_id: {[floats[i * 5] for i in range(10)]}")

# 检查x_center是否在0-1范围内
x_centers = floats[1::5]
valid_x = sum(1 for x in x_centers if 0 <= x <= 1)
print(f"  有效的x_center (0-1): {valid_x}/{len(x_centers)}")

# 检查是否有NaN
import math

nan_count = sum(1 for f in floats if math.isnan(f))
print(f"  NaN数量: {nan_count}")
