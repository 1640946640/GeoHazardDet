import os

# 检查label目录下有哪些格式的文件
label_dir = 'datasets/label'
all_files = os.listdir(label_dir)
tif_count = sum(1 for f in all_files if f.endswith('.tif'))
txt_count = sum(1 for f in all_files if f.endswith('.txt'))

print(f'label目录总文件数: {len(all_files)}')
print(f'.tif文件数: {tif_count}')
print(f'.txt文件数: {txt_count}')

# 检查是否有其他格式的文件
other = [f for f in all_files if not f.endswith('.tif') and not f.endswith('.txt')]
print(f'其他格式文件: {other[:10]}')  # 显示前10个

# 检查几个tif标签文件的详细信息
import struct
for f in ['moxizheng_0.2m_UAV0069.tif', 'moxizheng_0.2m_UAV1680.tif']:
    path = os.path.join(label_dir, f)
    size = os.path.getsize(path)
    print(f'\n{f}:')
    print(f'  大小: {size} bytes')
    
    # 检查是否是有效的YOLO格式 (每目标20字节)
    if size % 20 == 0:
        num_boxes = size // 20
        print(f'  可能是YOLO格式: {num_boxes} 个目标')
    else:
        print(f'  不是标准YOLO格式 (20字节倍数)')
        # 检查是否是图片格式
        if size == 512 * 512 * 3 + 8:  # 512x512 RGB + 8字节头
            print(f'  可能是带头的图片格式')
        elif size == 512 * 512:
            print(f'  可能是512x512灰度图')
        elif size == 512 * 512 * 4:
            print(f'  可能是512x512 RGBA')
