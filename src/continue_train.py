# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  继续训练脚本 - 基于已有模型继续训练
# ============================================================
#
# 【功能】
# - 基于之前训练好的模型继续训练
# - 降低学习率进行微调
# - 模型保存到项目目录
#
# 【使用方法】
# # 继续训练50轮
# python src/continue_train.py --weights models/train_results/exp/weights/best.pt --epochs 50
#
# # 继续训练100轮，使用较小batch
# python src/continue_train.py --weights models/train_results/exp/weights/best.pt --epochs 100 --batch 8
#
# ============================================================

import sys
from pathlib import Path


def continue_train(
    weights: str = "models/train_results/exp/weights/best.pt",
    data: str = "configs/disaster.yaml",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "models/train_results",
    name: str = "exp_v2",
    optimizer: str = "Adam",
    lr0: float = 0.001,
    lrf: float = 0.01,
    cos_lr: bool = True,
    close_mosaic: int = 20,
    save_period: int = 10,
    patience: int = 100,
    amp: bool = True,
) -> None:
    """基于已有模型继续训练.

    Args:
        weights: 之前训练的模型权重路径
        data: 数据集配置文件路径
        epochs: 继续训练的轮次
        batch: 批次大小
        imgsz: 输入图片尺寸
        device: 训练设备
        project: 项目保存目录（项目内相对路径）
        name: 实验名称
        optimizer: 优化器类型
        lr0: 初始学习率（较低，用于微调）
        lrf: 最终学习率因子
        cos_lr: 是否使用余弦学习率
        close_mosaic: 最后N轮关闭mosaic增强
        save_period: 模型保存间隔
        patience: 早停耐心值
        amp: 是否使用自动混合精度
    """
    # 导入ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics库")
        print("请运行: pip install ultralytics>=8.1.0")
        sys.exit(1)

    # 检查权重文件
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"错误: 权重文件不存在: {weights}")
        print("请先运行训练: python src/train.py")
        sys.exit(1)

    print("=" * 60)
    print("泥石流/滑坡地质灾害识别模型 - 继续训练")
    print("=" * 60)
    print(f"\n基础模型: {weights}")
    print(f"继续训练: {epochs} 轮")
    print(f"批次大小: {batch}")
    print(f"图片尺寸: {imgsz}")
    print(f"初始学习率: {lr0}")
    print(f"优化器: {optimizer}")
    print(f"保存目录: {project}/{name}")
    print("=" * 60)

    # 1. 加载之前训练的模型
    print("\n[1/3] 加载模型...")
    # 不使用预训练权重，直接加载用户指定的权重
    model = YOLO(weights)

    # 2. 继续训练
    print("\n[2/3] 开始训练...")
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        pretrained=False,  # 不使用预训练权重
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        save_period=save_period,
        patience=patience,
        amp=amp,
        verbose=True,
    )

    # 3. 输出最终指标
    print("\n[3/3] 训练完成！最终指标:")
    print(f"  - mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  - mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"  - Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"  - Recall: {results.results_dict['metrics/recall(B)']:.4f}")

    # 模型保存路径
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    last_weights = Path(results.save_dir) / "weights" / "last.pt"

    print(f"\n最佳模型: {best_weights}")
    print(f"最终模型: {last_weights}")
    print(f"\n模型已保存到项目目录: {Path(results.save_dir).relative_to(Path.cwd())}")


def parse_args():
    """解析命令行参数."""
    import argparse

    parser = argparse.ArgumentParser(description="泥石流/滑坡地质灾害识别模型继续训练")

    # 模型配置
    parser.add_argument(
        "--weights", "-w", type=str, default="models/train_results/exp/weights/best.pt", help="之前训练的模型权重路径"
    )

    # 数据配置
    parser.add_argument("--data", "-d", type=str, default="configs/disaster.yaml", help="数据集配置文件路径")

    # 训练参数
    parser.add_argument("--epochs", "-e", type=int, default=50, help="继续训练的轮次")
    parser.add_argument("--batch", "-b", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", "-img", type=int, default=640, help="输入图片尺寸")

    # 设备配置
    parser.add_argument("--device", "-dev", type=str, default="0", help="训练设备 (GPU编号或cpu)")

    # 输出配置
    parser.add_argument(
        "--project", "-p", type=str, default="models/train_results", help="项目保存目录（项目内相对路径）"
    )
    parser.add_argument("--name", "-n", type=str, default="exp_v2", help="实验名称")

    # 优化器配置
    parser.add_argument("--optimizer", type=str, default="Adam", help="优化器类型 (SGD/Adam/AdamW/auto)")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始学习率（建议使用较小值，如0.001）")
    parser.add_argument("--lrf", type=float, default=0.01, help="最终学习率因子")
    parser.add_argument("--cos_lr", action="store_true", default=True, help="使用余弦学习率")

    # 训练策略
    parser.add_argument("--patience", "-pat", type=int, default=100, help="早停耐心值")
    parser.add_argument("--save_period", type=int, default=10, help="模型保存间隔")
    parser.add_argument("--close_mosaic", type=int, default=20, help="最后N轮关闭mosaic增强")
    parser.add_argument("--amp", action="store_true", default=True, help="使用自动混合精度训练")

    return parser.parse_args()


def main():
    """主函数."""
    args = parse_args()

    # 转换设备参数
    if args.device == "cpu":
        device = "cpu"
    else:
        device = args.device

    # 开始训练
    continue_train(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        patience=args.patience,
        save_period=args.save_period,
        close_mosaic=args.close_mosaic,
        amp=args.amp,
    )


if __name__ == "__main__":
    main()
