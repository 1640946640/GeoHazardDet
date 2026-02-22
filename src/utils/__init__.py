# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  工具模块 - 日志与可视化
# ============================================================
#
# 【模块功能】
# - 提供日志记录功能
# - 支持结果可视化与保存
#
# 【主要类】
# - Logger: 日志记录器
# - Visualizer: 结果可视化工具
#
# 【使用示例】
# ```python
# from src.utils import Logger, Visualizer
#
# # 初始化日志
# logger = Logger(log_dir='logs/')
# logger.info('训练开始')
#
# # 初始化可视化
# viz = Visualizer()
# viz.plot_results(results)
# ```
#
# ============================================================
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Logger:
    """日志记录器.

    提供训练过程日志记录功能，支持控制台和文件输出

    【日志级别】
    - DEBUG: 调试信息
    - INFO: 一般信息
    - WARNING: 警告
    - ERROR: 错误
    - CRITICAL: 严重错误
    """

    def __init__(
        self,
        log_dir: str = "logs/",
        log_file: str | None = None,
        level: int = logging.INFO,
        console: bool = True,
    ):
        """初始化日志记录器.

        Args:
            log_dir: 日志目录
            log_file: 日志文件名，默认自动生成
            level: 日志级别
            console: 是否输出到控制台
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成日志文件名
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"train_{timestamp}.log"

        self.log_file = self.log_dir / log_file

        # 配置日志
        self.logger = logging.getLogger("GeoHazardDet")
        self.logger.setLevel(level)

        # 清除已有处理器
        self.logger.handlers.clear()

        # 文件处理器
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(level)

        # 格式化器
        formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message: str):
        """记录调试信息."""
        self.logger.debug(message)

    def info(self, message: str):
        """记录一般信息."""
        self.logger.info(message)

    def warning(self, message: str):
        """记录警告信息."""
        self.logger.warning(message)

    def error(self, message: str):
        """记录错误信息."""
        self.logger.error(message)

    def critical(self, message: str):
        """记录严重错误信息."""
        self.logger.critical(message)

    def log_metrics(self, metrics: dict[str, float], epoch: int, phase: str = "train"):
        """记录训练指标.

        Args:
            metrics: 指标字典
            epoch: 当前轮次
            phase: 阶段（train/val/test）
        """
        self.info(
            f"\
[{phase.upper()}] Epoch {epoch}"
        )
        for metric_name, value in metrics.items():
            self.info(f"  - {metric_name}: {value:.4f}")


class Visualizer:
    """结果可视化工具.

    提供训练曲线、检测结果等可视化功能

    【功能】
    - 绘制训练曲线
    - 可视化检测结果
    - 保存可视化图片
    """

    # 类别颜色配置
    COLORS = {
        0: (255, 0, 0),  # 泥石流 - 红色
        1: (0, 255, 0),  # 滑坡 - 绿色
    }

    # 类别名称
    CLASS_NAMES = {
        0: "泥石流",
        1: "滑坡",
    }

    def __init__(
        self,
        save_dir: str = "runs/visualize/",
        dpi: int = 150,
    ):
        """初始化可视化工具.

        Args:
            save_dir: 保存目录
            dpi: 图片分辨率
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_training_curves(
        self,
        epochs: list[int],
        metrics: dict[str, list[float]],
        save_path: str | None = None,
        title: str = "Training Curves",
    ):
        """绘制训练曲线.

        Args:
            epochs: 轮次列表
            metrics: 指标字典 {指标名: [值列表]}
            save_path: 保存路径
            title: 图表标题
        """
        plt.figure(figsize=(12, 8))

        for metric_name, values in metrics.items():
            plt.plot(epochs, values, label=metric_name, linewidth=2)

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"训练曲线已保存: {save_path}")

        plt.close()

    def visualize_detection(
        self,
        image: np.ndarray,
        detections: np.ndarray,
        class_names: dict[int, str] | None = None,
        conf_threshold: float = 0.5,
        save_path: str | None = None,
    ):
        """可视化检测结果.

        Args:
            image: 输入图片 (H, W, C) RGB格式
            detections: 检测结果数组 (N, 6) [x1, y1, x2, y2, conf, class_id]
            class_names: 类别名称字典
            conf_threshold: 置信度阈值
            save_path: 保存路径
        """
        if class_names is None:
            class_names = self.CLASS_NAMES

        # 复制图片
        image_copy = image.copy()

        # 转换BGR格式（如果是OpenCV图片）
        if image_copy.shape[2] == 3 and image_copy.dtype == np.uint8:
            image_copy = image_copy[:, :, ::-1]  # RGB to BGR

        # 转换为PIL Image
        pil_image = Image.fromarray(image_copy.astype(np.uint8))
        draw = ImageDraw.Draw(pil_image)

        # 获取图片尺寸
        _img_width, _img_height = pil_image.size

        # 遍历检测结果
        for det in detections:
            if len(det) < 6:
                continue

            x1, y1, x2, y2, conf, cls_id = det

            # 过滤低置信度
            if conf < conf_threshold:
                continue

            # 获取类别信息
            cls_id = int(cls_id)
            color = self.COLORS.get(cls_id, (255, 255, 0))
            class_name = class_names.get(cls_id, f"Class_{cls_id}")

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # 绘制标签
            label = f"{class_name}: {conf:.2f}"
            text_width, text_height = draw.textsize(label) if hasattr(draw, "textsize") else (100, 20)

            # 标签背景
            draw.rectangle([x1, y1 - text_height - 5, x1 + text_width + 5, y1], fill=color)

            # 标签文字
            draw.text((x1 + 3, y1 - text_height - 3), label, fill=(255, 255, 255))

        # 保存结果
        if save_path:
            pil_image.save(save_path)
            print(f"检测结果已保存: {save_path}")

        return np.array(pil_image)

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: list[str] | None = None,
        save_path: str | None = None,
        normalize: bool = True,
    ):
        """绘制混淆矩阵.

        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称列表
            save_path: 保存路径
            normalize: 是否归一化
        """
        if class_names is None:
            class_names = ["debris_flow", "landslide"]

        if normalize:
            cm = confusion_matrix.astype("float") / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        else:
            cm = confusion_matrix

        plt.figure(figsize=(8, 6))

        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")

        # 添加颜色条
        plt.colorbar()

        # 设置轴标签
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # 在格子中添加数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black"
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"混淆矩阵已保存: {save_path}")

        plt.close()

    def plot_class_distribution(
        self,
        class_counts: dict[str, int],
        save_path: str | None = None,
    ):
        """绘制类别分布图.

        Args:
            class_counts: 类别计数字典
            save_path: 保存路径
        """
        plt.figure(figsize=(8, 6))

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        colors = ["#FF6B6B", "#4ECDC4"]  # 红色-泥石流, 青色-滑坡

        plt.bar(classes, counts, color=colors, edgecolor="black", linewidth=1.5)

        plt.xlabel("Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title("Class Distribution", fontsize=14)

        # 添加数值标签
        for i, (cls, cnt) in enumerate(zip(classes, counts)):
            plt.text(i, cnt + max(counts) * 0.02, str(cnt), ha="center", fontsize=12, fontweight="bold")

        plt.ylim(0, max(counts) * 1.15)
        plt.grid(axis="y", alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"类别分布图已保存: {save_path}")

        plt.close()
