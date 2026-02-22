import numpy as np

label_path = "datasets/label/moxizheng_0.2m_UAV0069.tif"

# 读取文件
with open(label_path, "rb") as f:
    data = f.read()

print(f"文件大小: {len(data)} bytes")

# 检查文件头 (通常是8-16字节)
header_size = 8  # 常见的头大小
header = data[:header_size]
print(f"文件头 ({header_size} bytes): {header.hex()}")

# 检查是否是标准TIFF格式
if data[:4] == b"II\x2a\x00" or data[:4] == b"MM\x00\x2a":
    print("文件头显示这是标准TIFF格式")
else:
    print("不是标准TIFF格式，可能是自定义格式")

# 检查是否包含有效的class_id
# 假设每目标5个float32 = 20字节
# 或者可能是每像素1字节的分割掩码

# 方案1: 假设是分割掩码 (每像素1字节，值=类别ID)
data_1byte = data  # 整个文件作为字节数组
unique_values = set(data_1byte)
print("\n分割掩码检查:")
print(f"  唯一值数量: {len(unique_values)}")
print(f"  值范围: {min(data_1byte)} - {max(data_1byte)}")
print("  值频率(0,1,2,...):")
for v in sorted(unique_values)[:10]:
    count = data_1byte.count(v)
    print(f"    {v}: {count} 次")

# 方案2: 假设是float32数组
print("\nFloat32数组检查:")
floats = np.frombuffer(data, dtype=np.float32)
print(f"  总float数: {len(floats)}")
print(f"  值范围: {floats.min():.4f} - {floats.max():.4f}")
print(f"  NaN数量: {np.isnan(floats).sum()}")
print(f"  Inf数量: {np.isinf(floats).sum()}")

# 有效值检查 (class_id应该是0或1)
valid_masks = ~np.isnan(floats) & ~np.isinf(floats)
valid_floats = floats[valid_masks]
print(f"  有效值数量: {len(valid_floats)}")

# 检查class_id (每隔5个值)
if len(valid_floats) >= 5:
    class_ids = valid_floats[::5]
    unique_classes = set(class_ids)
    print(f"  唯一class_id: {sorted(unique_classes)}")
    for cls in sorted(unique_classes)[:5]:
        count = (class_ids == cls).sum()
        print(f"    class_id={cls}: {count} 次")

# 检查图片尺寸对应关系
img_size = 512 * 512
if len(floats) == img_size:
    print("\n可能是512x512分割掩码 (每像素1个float32)")
elif len(data) == img_size + 8:
    print("\n可能是512x512 + 8字节头")
elif len(data) == img_size * 4 + 8:
    print("\n可能是512x512 float32格式 + 8字节头")

# 尝试查找边界框数量
print("\n边界框分析:")
print(f"  有效值数量 / 5 = {len(valid_floats) // 5} 个可能的目标")
print(f"  有效值数量 / 4 = {len(valid_floats) // 4} 个可能的点")

# 保存一个样本供检查
print("\n前100个有效float值:")
count = 0
for i, f in enumerate(valid_floats):
    if count >= 20:
        break
    print(f"  [{i}] {f}")
    count += 1
