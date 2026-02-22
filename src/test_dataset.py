# ============================================================
#  地质灾害数据集测试脚本
#  测试数据集格式并生成标注可视化图片
# ============================================================
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class DisasterDatasetTester:
    """地质灾害数据集测试器.

    用于测试数据集格式、验证标注、生成可视化
    """

    # 类别映射
    CLASS_NAMES = {
        0: "泥石流",
        1: "滑坡",
    }

    CLASS_COLORS = {
        0: "red",
        1: "green",
    }

    def __init__(
        self,
        img_dir: str = "datasets/img",
        label_dir: str = "datasets/label",
        mask_dir: str = "datasets/mask",
    ):
        """初始化测试器.

        Args:
            img_dir: 图片目录
            label_dir: 标注目录
            mask_dir: 掩膜目录
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.mask_dir = Path(mask_dir)

        # 验证目录存在
        if not self.img_dir.exists():
            raise FileNotFoundError(f"图片目录不存在: {self.img_dir}")

        print("数据集路径:")
        print(f"  - 图片目录: {self.img_dir}")
        print(f"  - 标注目录: {self.label_dir}")
        print(f"  - 掩膜目录: {self.mask_dir}")

    def scan_dataset(self) -> dict:
        """扫描数据集，统计信息."""
        print("\n" + "=" * 60)
        print("数据集扫描")
        print("=" * 60)

        # 获取所有图片文件
        img_files = list(self.img_dir.glob("*.tif")) + list(self.img_dir.glob("*.tiff"))
        label_files = list(self.label_dir.glob("*.tif")) + list(self.label_dir.glob("*.tiff"))
        mask_files = list(self.mask_dir.glob("*.tif")) + list(self.mask_dir.glob("*.tiff"))

        print("\n文件统计:")
        print(f"  - 图片文件: {len(img_files)} 个")
        print(f"  - 标注文件: {len(label_files)} 个")
        print(f"  - 掩膜文件: {len(mask_files)} 个")

        # 获取文件名列表
        img_names = {f.stem for f in img_files}
        label_names = {f.stem for f in label_files}
        mask_names = {f.stem for f in mask_files}

        print("\n文件匹配:")
        print(f"  - 有标注的图片: {len(img_names & label_names)} 个")
        print(f"  - 有掩膜的图片: {len(img_names & mask_names)} 个")
        print(f"  - 完整数据（三者皆有）: {len(img_names & label_names & mask_names)} 个")

        # 统计信息
        stats = {
            "total_images": len(img_files),
            "total_labels": len(label_files),
            "total_masks": len(mask_files),
            "matched_images": len(img_names & label_names),
            "matched_masks": len(img_names & mask_names),
            "complete_samples": len(img_names & label_names & mask_names),
        }

        return stats

    def read_tiff_image(self, filepath: str) -> np.ndarray:
        """读取TIFF图片.

        Args:
            filepath: 文件路径

        Returns:
            numpy数组 (H, W, C) 或 (H, W)
        """
        try:
            img = Image.open(filepath)
            img_array = np.array(img)
            return img_array
        except Exception as e:
            print(f"读取图片失败: {filepath}, 错误: {e}")
            return None

    def analyze_label_format(self, sample_name: str) -> dict:
        """分析标注文件格式.

        Args:
            sample_name: 样本名称

        Returns:
            格式分析结果
        """
        label_path = self.label_dir / f"{sample_name}.tif"

        if not label_path.exists():
            return {"error": "标注文件不存在"}

        label_array = self.read_tiff_image(str(label_path))

        if label_array is None:
            return {"error": "无法读取标注"}

        info = {
            "shape": label_array.shape,
            "dtype": str(label_array.dtype),
            "min_value": float(np.min(label_array)),
            "max_value": float(np.max(label_array)),
            "unique_values": list(np.unique(label_array))[:20],  # 最多显示20个唯一值
        }

        return info

    def analyze_image(self, sample_name: str) -> dict:
        """分析图片文件格式.

        Args:
            sample_name: 样本名称

        Returns:
            格式分析结果
        """
        img_path = self.img_dir / f"{sample_name}.tif"

        if not img_path.exists():
            return {"error": "图片文件不存在"}

        img_array = self.read_tiff_image(str(img_path))

        if img_array is None:
            return {"error": "无法读取图片"}

        info = {
            "shape": img_array.shape,
            "dtype": str(img_array.dtype),
            "min_value": float(np.min(img_array)),
            "max_value": float(np.max(img_array)),
            "channels": img_array.shape[-1] if len(img_array.shape) > 2 else 1,
        }

        return info

    def analyze_mask(self, sample_name: str) -> dict:
        """分析掩膜文件格式.

        Args:
            sample_name: 样本名称

        Returns:
            格式分析结果
        """
        mask_path = self.mask_dir / f"{sample_name}.tif"

        if not mask_path.exists():
            return {"error": "掩膜文件不存在"}

        mask_array = self.read_tiff_image(str(mask_path))

        if mask_array is None:
            return {"error": "无法读取掩膜"}

        info = {
            "shape": mask_array.shape,
            "dtype": str(mask_array.dtype),
            "min_value": float(np.min(mask_array)),
            "max_value": float(np.max(mask_array)),
            "unique_values": list(np.unique(mask_array))[:20],
        }

        return info

    def visualize_sample(
        self,
        sample_name: str,
        output_path: str | None = None,
        show_label: bool = True,
        show_mask: bool = True,
    ):
        """可视化单个样本.

        Args:
            sample_name: 样本名称
            output_path: 输出路径
            show_label: 是否显示标注
            show_mask: 是否显示掩膜
        """
        print(f"\n可视化样本: {sample_name}")

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 原始图片
        img_path = self.img_dir / f"{sample_name}.tif"
        if img_path.exists():
            img_array = self.read_tiff_image(str(img_path))

            # 处理不同通道数
            if len(img_array.shape) == 2:
                axes[0].imshow(img_array, cmap="gray")
            elif img_array.shape[2] == 1:
                axes[0].imshow(img_array[:, :, 0], cmap="gray")
            elif img_array.shape[2] == 3:
                axes[0].imshow(img_array)
            else:
                # 显示第一波段
                axes[0].imshow(img_array[:, :, 0], cmap="gray")

            axes[0].set_title(f"原始图片\n{img_array.shape}", fontsize=12)
            axes[0].axis("off")
        else:
            axes[0].set_title("图片不存在", fontsize=12)

        # 2. 标注/掩膜
        if show_mask:
            mask_path = self.mask_dir / f"{sample_name}.tif"
            mask_array = None
            if mask_path.exists():
                mask_array = self.read_tiff_image(str(mask_path))
                axes[1].imshow(mask_array, cmap="viridis")
                axes[1].set_title(f"掩膜标注\n{mask_array.shape}", fontsize=12)
            else:
                axes[1].set_title("掩膜不存在", fontsize=12)
            axes[1].axis("off")
        else:
            label_path = self.label_dir / f"{sample_name}.tif"
            if label_path.exists():
                label_array = self.read_tiff_image(str(label_path))
                axes[1].imshow(label_array, cmap="hot")
                axes[1].set_title(f"标注数据\n{label_array.shape}", fontsize=12)
            axes[1].axis("off")

        # 3. 合并可视化
        if img_path.exists() and "mask_array" in dir() and mask_array is not None:
            # 创建RGBA叠加
            if len(img_array.shape) == 2:
                # 灰度图转RGB
                img_rgb = np.stack([img_array] * 3, axis=2)
            elif img_array.shape[2] == 1:
                img_rgb = np.concatenate([img_array] * 3, axis=2)
            else:
                img_rgb = img_array[:, :, :3]

            # 归一化
            img_norm = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-10)

            # 创建掩膜颜色
            mask_colored = np.zeros((*mask_array.shape, 4))

            # 泥石流（0）= 红色，滑坡（1）= 绿色
            mask_colored[mask_array == 0] = [1, 0, 0, 0.5]  # 红色半透明
            mask_colored[mask_array == 1] = [0, 1, 0, 0.5]  # 绿色半透明

            # 叠加
            axes[2].imshow(img_norm)
            axes[2].imshow(mask_colored)
            axes[2].set_title("标注叠加", fontsize=12)
        else:
            axes[2].set_title("无法生成叠加", fontsize=12)

        axes[2].axis("off")

        # 添加图例
        legend_elements = [
            patches.Patch(facecolor="red", alpha=0.5, label="泥石流 (0)"),
            patches.Patch(facecolor="green", alpha=0.5, label="滑坡 (1)"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.suptitle(f"地质灾害数据集样本: {sample_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"可视化结果已保存: {output_path}")
        else:
            plt.savefig("dataset_sample_visualization.png", dpi=150, bbox_inches="tight")
            print("可视化结果已保存: dataset_sample_visualization.png")

        plt.close()

    def test_samples(self, num_samples: int = 5) -> list[dict]:
        """测试多个样本.

        Args:
            num_samples: 测试样本数量

        Returns:
            测试结果列表
        """
        print("\n" + "=" * 60)
        print("样本测试")
        print("=" * 60)

        # 获取所有样本
        img_files = list(self.img_dir.glob("*.tif"))
        sample_names = [f.stem for f in img_files]

        # 随机选择样本
        import random

        random.seed(42)
        test_samples = random.sample(sample_names, min(num_samples, len(sample_names)))

        results = []
        for i, sample_name in enumerate(test_samples, 1):
            print(f"\n[{i}/{len(test_samples)}] 测试样本: {sample_name}")

            result = {
                "sample_name": sample_name,
                "image_info": self.analyze_image(sample_name),
                "label_info": self.analyze_label_format(sample_name),
                "mask_info": self.analyze_mask(sample_name),
            }

            results.append(result)

            # 打印摘要
            print(f"  图片: {result['image_info']}")
            print(f"  标注: {result['label_info']}")
            print(f"  掩膜: {result['mask_info']}")

        return results

    def convert_labels_to_yolo_format(self, output_dir: str = "datasets/labels/"):
        """将掩膜/标注转换为YOLO格式.

        Args:
            output_dir: 输出目录
        """
        print("\n" + "=" * 60)
        print("转换标签格式")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有掩膜文件
        mask_files = list(self.mask_dir.glob("*.tif"))

        print(f"找到 {len(mask_files)} 个掩膜文件")

        converted_count = 0
        for mask_path in mask_files:
            sample_name = mask_path.stem

            # 读取掩膜
            mask_array = self.read_tiff_image(str(mask_path))

            if mask_array is None:
                continue

            # 提取边界框
            boxes = self.mask_to_boxes(mask_array)

            # 保存YOLO格式
            label_path = output_path / f"{sample_name}.txt"

            with open(label_path, "w") as f:
                for box in boxes:
                    class_id, x_center, y_center, width, height = box
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            converted_count += 1

        print(f"成功转换 {converted_count} 个标签文件")
        print(f"输出目录: {output_path}")

        return output_path

    def mask_to_boxes(self, mask_array: np.ndarray) -> list[tuple]:
        """将掩膜转换为边界框.

        Args:
            mask_array: 掩膜数组

        Returns:
            边界框列表 [(class_id, x_center, y_center, width, height), ...]
        """
        boxes = []

        # 获取唯一值（排除0）
        unique_values = np.unique(mask_array)
        unique_values = unique_values[unique_values != 0]

        for class_id in unique_values:
            # 创建二值掩膜
            binary_mask = mask_array == class_id

            if not binary_mask.any():
                continue

            # 获取边界框坐标
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # 转换为YOLO格式（归一化）
            h, w = mask_array.shape

            x_center = (cmin + cmax) / 2 / w
            y_center = (rmin + rmax) / 2 / h
            width = (cmax - cmin) / w
            height = (rmax - rmin) / h

            boxes.append((int(class_id), x_center, y_center, width, height))

        return boxes


def main():
    """主函数."""
    # 初始化测试器
    tester = DisasterDatasetTester(
        img_dir="datasets/img",
        label_dir="datasets/label",
        mask_dir="datasets/mask",
    )

    # 1. 扫描数据集
    stats = tester.scan_dataset()

    # 2. 测试样本
    results = tester.test_samples(num_samples=3)

    # 3. 选择一个样本进行可视化
    if results:
        sample_name = results[0]["sample_name"]
        tester.visualize_sample(
            sample_name,
            output_path=f"datasets/{sample_name}_visualization.png",
            show_mask=True,
        )

    # 4. 转换标签格式
    tester.convert_labels_to_yolo_format()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

    return stats, results


if __name__ == "__main__":
    main()
