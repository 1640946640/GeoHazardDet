# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  自定义数据集类 - 适配YOLO格式
# ============================================================
#
# 【模块功能】
# - 加载YOLO格式的图像和标注
# - 支持数据增强和预处理
# - 与PyTorch DataLoader兼容
#
# 【数据格式】
# - 图片: 支持JPG/PNG格式
# - 标注: YOLO格式txt文件
#
# ============================================================
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DisasterDataset(Dataset):
    """地质灾害数据集类.

    用于加载YOLO格式的泥石流/滑坡检测数据集

    【数据集结构】 datasets/
    ├── images/
    │   ├── train/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        │   ├── 000001.txt
        │   └── ...
        ├── val/
        └── test/

    【标注格式】 每行一个目标: <class_id> <x_center> <y_center> <width> <height> 坐标均为归一化值（0-1范围）

    【使用示例】
    ```python
    from src.data import DisasterDataset

    # 创建数据集
    train_dataset = DisasterDataset(
        img_dir="datasets/images/train",
        label_dir="datasets/labels/train",
        class_names=["debris_flow", "landslide"],
        transform=transforms.Compose([...]),
    )

    # 使用DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    ```
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        class_names: list[str] | None = None,
        transform: object | None = None,
        img_size: int = 640,
    ):
        """初始化数据集.

        Args:
            img_dir: 图片目录路径
            label_dir: 标注目录路径
            class_names: 类别名称列表
            transform: 数据增强/预处理变换
            img_size: 图片目标尺寸
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.class_names = class_names or ["debris_flow", "landslide"]
        self.transform = transform
        self.img_size = img_size

        # 获取图片文件列表
        self.img_files = self._get_image_files()

        # 类别数量
        self.num_classes = len(self.class_names)

    def _get_image_files(self) -> list[Path]:
        """获取目录下所有图片文件.

        Returns:
            图片文件路径列表
        """
        # 支持的图片格式
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        img_files = []
        for ext in img_extensions:
            img_files.extend(self.img_dir.glob(f"*{ext}"))
            img_files.extend(self.img_dir.glob(f"*{ext.upper()}"))

        # 排序以保证顺序一致
        img_files = sorted(img_files)

        if not img_files:
            print(f"警告: 在 {self.img_dir} 中未找到图片文件")

        return img_files

    def _load_label(self, label_path: Path) -> np.ndarray:
        """加载YOLO格式标签文件.

        Args:
            label_path: 标签文件路径

        Returns:
            标签数组，形状为(n, 5)，每行[class_id, x, y, w, h]
        """
        if not label_path.exists():
            # 如果标签文件不存在，返回空数组
            return np.zeros((0, 5), dtype=np.float32)

        try:
            # 读取标签文件
            with open(label_path) as f:
                lines = f.readlines()

            # 解析标签
            labels = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                values = line.split()
                if len(values) >= 5:
                    class_id = int(values[0])
                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                    labels.append([class_id, x_center, y_center, width, height])

            return np.array(labels, dtype=np.float32)

        except Exception as e:
            print(f"加载标签文件失败 {label_path}: {e}")
            return np.zeros((0, 5), dtype=np.float32)

    def _load_image(self, img_path: Path) -> Image.Image:
        """加载图片.

        Args:
            img_path: 图片文件路径

        Returns:
            PIL Image对象
        """
        try:
            image = Image.open(img_path).convert("RGB")
            return image
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}")
            # 返回空白图片
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

    def __len__(self) -> int:
        """返回数据集样本数量.

        Returns:
            样本数量
        """
        return len(self.img_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray, str]:
        """获取单个样本.

        Args:
            idx: 样本索引

        Returns:
            - image: 图片张量 (C, H, W)
            - labels: 标签数组 (N, 5)
            - img_path: 图片路径字符串
        """
        # 获取图片路径
        img_path = self.img_files[idx]

        # 加载图片
        image = self._load_image(img_path)

        # 获取标签路径
        # 假设标签文件名与图片文件名相同，只是扩展名不同
        label_path = self.label_dir / (img_path.stem + ".txt")

        # 加载标签
        labels = self._load_label(label_path)

        # 应用数据增强/预处理
        if self.transform is not None:
            image = self.transform(image)

        return image, labels, str(img_path)

    def get_class_stats(self) -> dict[str, int]:
        """统计各类别样本数量.

        Returns:
            类别统计字典 {类别名: 样本数}
        """
        stats = {name: 0 for name in self.class_names}

        for img_path in self.img_files:
            label_path = self.label_dir / (img_path.stem + ".txt")
            labels = self._load_label(label_path)

            for label in labels:
                class_id = int(label[0])
                if 0 <= class_id < len(self.class_names):
                    stats[self.class_names[class_id]] += 1

        return stats


def collate_fn(batch):
    """自定义批处理函数.

    用于DataLoader对不同大小的标签进行批处理

    Args:
        batch: 批次数据

    Returns:
        - images: 图片张量堆叠 (B, C, H, W)
        - labels: 标签列表，每个元素为(N, 5)数组
        - paths: 图片路径列表
    """
    images = []
    labels = []
    paths = []

    for img, label, path in batch:
        images.append(img)
        labels.append(label)
        paths.append(path)

    # 堆叠图片张量
    images = torch.stack(images, dim=0)

    return images, labels, paths


def create_datasets(
    data_dir: str = "datasets/",
    img_size: int = 640,
    transform: object | None = None,
) -> tuple[DisasterDataset, DisasterDataset, DisasterDataset]:
    """创建训练集、验证集、测试集.

    Args:
        data_dir: 数据集根目录
        img_size: 图片目标尺寸
        transform: 数据变换

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = DisasterDataset(
        img_dir=os.path.join(data_dir, "images/train"),
        label_dir=os.path.join(data_dir, "labels/train"),
        class_names=["debris_flow", "landslide"],
        transform=transform,
        img_size=img_size,
    )

    val_dataset = DisasterDataset(
        img_dir=os.path.join(data_dir, "images/val"),
        label_dir=os.path.join(data_dir, "labels/val"),
        class_names=["debris_flow", "landslide"],
        transform=transform,
        img_size=img_size,
    )

    test_dataset = DisasterDataset(
        img_dir=os.path.join(data_dir, "images/test"),
        label_dir=os.path.join(data_dir, "labels/test"),
        class_names=["debris_flow", "landslide"],
        transform=transform,
        img_size=img_size,
    )

    return train_dataset, val_dataset, test_dataset
