import numpy as np
from PIL import Image

# 检查mask目录的文件
mask_path = "datasets/mask/moxizheng_0.2m_UAV1690.tif"
arr = np.array(Image.open(mask_path))
print("Mask analysis:")
print(f"  Shape: {arr.shape}")
print(f"  Unique values: {np.unique(arr)}")

# 检查label目录的另一个文件
label_path = "datasets/label/moxizheng_0.2m_UAV1680.tif"
arr2 = np.array(Image.open(label_path))
print("\nLabel analysis:")
print(f"  Shape: {arr2.shape}")

# 对比两个文件的唯一值
print("\nLabel unique RGB:")
if len(arr2.shape) == 3:
    unique_colors = set()
    for i in range(arr2.shape[0]):
        for j in range(arr2.shape[1]):
            unique_colors.add(tuple(arr2[i, j]))
    print(f"  Count: {len(unique_colors)}")

# 检查原图
img_path = "datasets/img/moxizheng_0.2m_UAV1680.tif"
img = np.array(Image.open(img_path))
print("\nOriginal image analysis:")
print(f"  Shape: {img.shape}")
print(f"  Value range: {img.min()} - {img.max()}")

# 看看label是否可能是原图的副本
if arr2.shape == img.shape:
    print("\nLabel have SAME shape!")
    if np.array_equal(arr2, img):
        print("ERROR: Label is IDENTICAL to Image!")
    else:
        diff = np.abs(arr2.astype(int) - img.astype(int))
        print(f"  Difference: min={diff.min()}, max={diff.max()}, mean={diff.mean():.2f}")
