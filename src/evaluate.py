# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  模型评估脚本
# ============================================================
#
# 【基于Ultralytics YOLOv8开发】
# 本项目基于Ultralytics YOLOv8框架开发，遵循AGPL-3.0许可证
# 参考: https://github.com/ultralytics/ultralytics
#
# 【功能】
# - 在测试集上评估模型性能
# - 计算各类评估指标（mAP, Precision, Recall, F1等）
# - 生成混淆矩阵
# - 输出评估报告
#
# 【使用方法】
# python src/evaluate.py  # 默认使用项目内best.pt
# python src/evaluate.py --weights models/weights/best.pt
# python src/evaluate.py --weights models/weights/last.pt --save-report models/my_report.txt
#
# ============================================================

import os
import sys
from pathlib import Path

import numpy as np


def get_available_device(requested_device: str = "0") -> str:
    """自动检测可用的评估设备.

    如果请求的设备不可用（如 NVIDIA GPU），自动回退到 CPU

    Args:
        requested_device: 请求的设备 (如 '0', 'cpu', '0,1')

    Returns:
        可用的设备字符串
    """
    import torch

    # 如果明确指定了 CPU，直接返回
    if requested_device.lower() == "cpu":
        return "cpu"

    # 如果指定了具体的 GPU 编号
    if requested_device.isdigit():
        device_id = int(requested_device)

        # 检查 CUDA 是否可用
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            return requested_device
        else:
            print(f"⚠️ 警告: 请求的 CUDA 设备 {device_id} 不可用")
            print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
            print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
            print("   自动切换到 CPU 运行")
            return "cpu"

    # 其他情况（如 '0,1' 多GPU），检查是否全部可用
    if "," in requested_device:
        available_devices = []
        for dev in requested_device.split(","):
            dev = dev.strip()
            if dev.isdigit() and torch.cuda.is_available() and int(dev) < torch.cuda.device_count():
                available_devices.append(dev)

        if available_devices:
            return ",".join(available_devices)
        else:
            print("⚠️ 警告: 请求的 CUDA 设备不可用，自动切换到 CPU 运行")
            return "cpu"

    # 默认返回原始请求
    return requested_device


class DisasterEvaluator:
    """地质灾害检测模型评估器.

    提供完整的模型评估功能

    【评估指标】
    - mAP@0.5: IoU阈值0.5下的平均精度
    - mAP@0.5:0.95: IoU阈值0.5-0.95的平均精度
    - Precision: 精确率
    - Recall: 召回率
    - F1-Score: F1分数
    - Confusion Matrix: 混淆矩阵
    """

    # 类别名称
    CLASS_NAMES = {
        0: "debris_flow",  # 泥石流
        1: "landslide",  # 滑坡
    }

    def __init__(
        self,
        weights: str,
        data: str,
        imgsz: int = 640,
        conf_thres: float = 0.001,
        iou_thres: float = 0.6,
        device: str = "0",
        save_dir: str = "runs/val/",
        verbose: bool = True,
    ):
        """初始化评估器.

        Args:
            weights: 模型权重文件路径
            data: 数据集配置文件路径
            imgsz: 输入图片尺寸
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            device: 评估设备
            save_dir: 结果保存目录
            verbose: 是否详细输出
        """
        self.weights = Path(weights)
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # 自动检测可用设备，如果GPU不可用则自动切换到CPU
        self.device = get_available_device(device)
        self.save_dir = Path(save_dir)
        self.verbose = verbose

        # 加载模型
        self.model = self._load_model()

    def _load_model(self):
        """加载YOLOv8模型."""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("错误: 未安装ultralytics库")
            print("请运行: pip install ultralytics>=8.1.0")
            sys.exit(1)

        if not self.weights.exists():
            print(f"错误: 模型文件不存在: {self.weights}")
            sys.exit(1)

        print(f"加载模型: {self.weights}")
        model = YOLO(str(self.weights))
        return model

    def evaluate(self) -> dict:
        """执行模型评估.

        Returns:
            评估结果字典
        """
        print("=" * 60)
        print("模型评估")
        print("=" * 60)

        # 使用ultralytics内置评估
        results = self.model.val(
            data=self.data,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            save_dir=self.save_dir,
            verbose=self.verbose,
        )

        # 提取关键指标
        metrics = {
            "mAP50": results.results_dict["metrics/mAP50(B)"],
            "mAP50_95": results.results_dict["metrics/mAP50-95(B)"],
            "precision": results.results_dict["metrics/precision(B)"],
            "recall": results.results_dict["metrics/recall(B)"],
            "f1": 2
            * (results.results_dict["metrics/precision(B)"] * results.results_dict["metrics/recall(B)"])
            / (results.results_dict["metrics/precision(B)"] + results.results_dict["metrics/recall(B)"] + 1e-10),
        }

        # 各类别指标 - 使用正确的API
        try:
            per_class_metrics = results.metrics.per_class_ap50()
            metrics["per_class_ap50"] = {
                self.CLASS_NAMES[i]: per_class_metrics[i] for i in range(len(self.CLASS_NAMES))
            }
        except AttributeError:
            # 如果API不可用，跳过类别指标
            metrics["per_class_ap50"] = {}

        return metrics

    def print_report(self, metrics: dict):
        """打印评估报告.

        Args:
            metrics: 评估指标字典
        """
        print("\n" + "=" * 60)
        print("评估报告")
        print("=" * 60)

        print("\n【总体指标】")
        print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1-Score:     {metrics['f1']:.4f}")

        print("\n【各类别AP@0.5】")
        for class_name, ap in metrics["per_class_ap50"].items():
            print(f"  {class_name}: {ap:.4f}")

        print("\n" + "=" * 60)

    def save_report(self, metrics: dict, save_path: str = "eval_report.txt"):
        """保存评估报告.

        Args:
            metrics: 评估指标字典
            save_path: 保存路径
        """
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("泥石流/滑坡地质灾害检测模型评估报告\n")
            f.write("=" * 60 + "\n\n")

            f.write("【总体指标】\n")
            f.write(f"  mAP@0.5:      {metrics['mAP50']:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics['mAP50_95']:.4f}\n")
            f.write(f"  Precision:    {metrics['precision']:.4f}\n")
            f.write(f"  Recall:       {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:     {metrics['f1']:.4f}\n\n")

            f.write("【各类别AP@0.5】\n")
            for class_name, ap in metrics["per_class_ap50"].items():
                f.write(f"  {class_name}: {ap:.4f}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"评估报告已保存: {save_path}")


def calculate_map(predictions: list, ground_truths: list, num_classes: int = 2) -> dict:
    """计算mAP指标.

    Args:
        predictions: 预测结果列表
        ground_truths: 真实标签列表
        num_classes: 类别数量

    Returns:
        包含各指标的字典
    """
    # 计算Precision和Recall
    # 这里提供简化版本，实际使用ultralytics内置函数更准确

    from collections import defaultdict

    # 按类别统计
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred in predictions:
        class_id = pred["class_id"]
        class_stats[class_id]["tp"] += 1

    for gt in ground_truths:
        class_id = gt["class_id"]
        class_stats[class_id]["fn"] += 1

    # 计算每类的Precision和Recall
    metrics = {}
    for class_id in range(num_classes):
        stats = class_stats[class_id]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        metrics[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # 计算平均值
    avg_precision = np.mean([m["precision"] for m in metrics.values()])
    avg_recall = np.mean([m["recall"] for m in metrics.values()])
    avg_f1 = np.mean([m["f1"] for m in metrics.values()])

    metrics["average"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }

    return metrics


def parse_args():
    """解析命令行参数."""
    import argparse

    parser = argparse.ArgumentParser(description="泥石流/滑坡地质灾害检测模型评估")

    # 模型配置
    parser.add_argument(
        "--weights", "-w", type=str, default="models/weights/best.pt", help="模型权重文件路径 (默认使用项目内best.pt)"
    )

    # 数据配置
    parser.add_argument("--data", "-d", type=str, default="configs/disaster.yaml", help="数据集配置文件路径")

    # 评估参数
    parser.add_argument("--imgsz", "-img", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="IoU阈值")
    parser.add_argument("--device", "-dev", type=str, default="0", help="评估设备")

    # 输出配置
    parser.add_argument(
        "--save-dir", "-p", type=str, default="models/val_results", help="结果保存目录 (项目内相对路径)"
    )
    parser.add_argument("--save-report", type=str, default="models/eval_report.txt", help="评估报告保存路径 (项目内)")

    return parser.parse_args()


def main():
    """主函数."""
    # 确保从项目目录运行
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    args = parse_args()

    # 初始化评估器
    evaluator = DisasterEvaluator(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        save_dir=args.save_dir,
    )

    # 执行评估
    metrics = evaluator.evaluate()

    # 打印报告
    evaluator.print_report(metrics)

    # 保存报告
    evaluator.save_report(metrics, args.save_report)

    return metrics


if __name__ == "__main__":
    main()
