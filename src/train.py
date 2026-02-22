# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  训练脚本 - 基于YOLOv8
# ============================================================
#
# 【基于Ultralytics YOLOv8开发】
# 本项目基于Ultralytics YOLOv8框架开发，遵循AGPL-3.0许可证
# 参考: https://github.com/ultralytics/ultralytics
#
# 【功能】
# - 使用YOLOv8训练地质灾害检测模型
# - 支持多尺度训练、可视化监控
# - 自动保存最佳模型
#
# 【使用方法】
# python src/train.py  # 默认使用项目内best.pt训练300轮
# python src/train.py --epochs 100 --batch 16
# python src/train.py --model yolov8n.pt --epochs 300  # 从头训练
#
# ============================================================

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml


def get_available_device(requested_device: str = '0') -> str:
    """
    自动检测可用的训练设备

    如果请求的设备不可用（如 NVIDIA GPU），自动回退到 CPU

    Args:
        requested_device: 请求的设备 (如 '0', 'cpu', '0,1')

    Returns:
        可用的设备字符串
    """
    import torch

    # 如果明确指定了 CPU，直接返回
    if requested_device.lower() == 'cpu':
        return 'cpu'

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
            print(f"   自动切换到 CPU 运行")
            return 'cpu'

    # 其他情况（如 '0,1' 多GPU），检查是否全部可用
    if ',' in requested_device:
        available_devices = []
        for dev in requested_device.split(','):
            dev = dev.strip()
            if dev.isdigit() and torch.cuda.is_available() and int(dev) < torch.cuda.device_count():
                available_devices.append(dev)

        if available_devices:
            return ','.join(available_devices)
        else:
            print(f"⚠️ 警告: 请求的 CUDA 设备不可用，自动切换到 CPU 运行")
            return 'cpu'

    # 默认返回原始请求
    return requested_device


def load_config(config_path: str) -> Dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def fix_dataset_path(config_path: str) -> str:
    """
    自动修复数据集配置文件中的路径

    当项目移动到其他位置时，配置文件中的绝对路径会失效
    此函数会自动检测并修复为正确的相对路径

    Args:
        config_path: 配置文件路径

    Returns:
        修复后的配置文件路径
    """
    import shutil

    # 获取当前工作目录（项目根目录）
    project_root = Path.cwd()
    datasets_path = project_root / 'datasets'

    # 检查数据集目录是否存在
    if not datasets_path.exists():
        print(f"⚠️ 警告: 数据集目录不存在: {datasets_path}")
        return config_path

    # 加载配置文件
    config = load_config(config_path)

    # 获取配置中的原始路径
    original_path = config.get('path', '')

    # 检查原始路径是否有效
    if original_path and Path(original_path).exists():
        # 路径有效，不需要修复
        return config_path

    # 路径无效，需要修复 - 使用相对路径
    print(f"⚠️ 警告: 数据集路径 {original_path} 无效")
    print(f"   自动修复为相对路径: {datasets_path}")

    # 创建修复后的配置文件（使用相对路径）
    config['path'] = str(datasets_path)

    # 保存到临时位置，使用相对路径
    fixed_config_path = project_root / 'configs' / 'disaster_fixed.yaml'

    with open(fixed_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"   已生成修复后的配置文件: {fixed_config_path}")

    return str(fixed_config_path)


def get_model_size(model_name: str = 'yolov8n.pt') -> str:
    """
    根据需求选择合适的模型尺寸
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型路径
    """
    # YOLOv8模型尺寸对比：
    # yolov8n.pt - 纳米级 (3.2M参数) - 最快，精度较低
    # yolov8s.pt - 小型 (11.2M参数) - 快速平衡
    # yolov8m.pt - 中型 (25.9M参数) - 精度速度平衡
    # yolov8l.pt - 大型 (43.7M参数) - 高精度
    # yolov8x.pt - 超大级 (68.2M参数) - 最高精度
    return model_name


def train(
    data: str = 'configs/disaster.yaml',
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    model: str = 'models/weights/best.pt',
    device: str = '0',
    project: str = 'models/train_results',  # 项目内相对路径
    name: str = 'exp',
    pretrained: bool = False,
    optimizer: str = 'auto',
    patience: int = 100,
    save_period: int = 10,
    amp: bool = True,
    lr0: float = 0.01,
    lrf: float = 0.01,
    cos_lr: bool = False,
    close_mosaic: int = 10,
    resume: bool = False,
    **kwargs,
) -> Dict:
    """
    训练YOLOv8模型
    
    Args:
        data: 数据集配置文件路径
        epochs: 训练轮次
        batch: 批次大小
        imgsz: 输入图片尺寸
        model: 预训练模型路径
        device: 训练设备 (如 '0' 表示GPU0, 'cpu' 表示CPU)
        project: 项目保存目录（相对于项目根目录）
        name: 实验名称
        pretrained: 是否使用预训练权重
        optimizer: 优化器类型
        patience: 早停耐心值
        save_period: 模型保存间隔
        amp: 是否使用自动混合精度
        lr0: 初始学习率
        lrf: 最终学习率（余弦退火）
        cos_lr: 是否使用余弦学习率
        close_mosaic: 最后N轮关闭mosaic增强
        resume: 是否恢复训练
        
    Returns:
        训练结果字典
    """
    # 导入ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("错误: 未安装ultralytics库")
        print("请运行: pip install ultralytics>=8.1.0")
        sys.exit(1)

    # 自动修复数据集路径（当项目移动到其他位置时）
    if data and data.endswith('.yaml'):
        data = fix_dataset_path(data)

    # 确保保存目录是绝对路径（项目内）
    import os
    if not os.path.isabs(project):
        project = str(Path.cwd() / project)
    # 创建保存目录
    Path(project).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("泥石流/滑坡地质灾害识别模型训练")
    print("=" * 60)
    print(f"\n训练配置:")
    print(f"  - 数据集: {data}")
    print(f"  - 训练轮次: {epochs}")
    print(f"  - 批次大小: {batch}")
    print(f"  - 图片尺寸: {imgsz}")
    print(f"  - 预训练模型: {model}")
    print(f"  - 训练设备: {device}")
    print(f"  - 保存目录: {project}/{name}")
    print("=" * 60)
    
    # 1. 初始化模型
    print("\n[1/4] 初始化模型...")
    yolo_model = YOLO(model)
    
    # 2. 训练模型
    print("\n[2/4] 开始训练...")
    results = yolo_model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        pretrained=pretrained,
        optimizer=optimizer,
        patience=patience,
        save_period=save_period,
        amp=amp,
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        resume=resume,
        verbose=True,
    )
    
    # 3. 保存训练结果
    print("\n[3/4] 保存训练结果...")
    results.save_dir = Path(results.save_dir)
    
    # 4. 输出最终指标
    print("\n[4/4] 训练完成！最终指标:")
    print(f"  - mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"  - mAP50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"  - Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"  - Recall: {results.results_dict['metrics/recall(B)']:.4f}")
    
    # 模型保存路径
    best_weights = results.save_dir / 'weights' / 'best.pt'
    last_weights = results.save_dir / 'weights' / 'last.pt'
    
    print(f"\n最佳模型: {best_weights}")
    print(f"最终模型: {last_weights}")
    
    return results.results_dict


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='泥石流/滑坡地质灾害识别模型训练'
    )
    
    # 数据配置
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='configs/disaster.yaml',
        help='数据集配置文件路径'
    )
    
    # 训练参数
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=300,
        help='训练轮次'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='批次大小'
    )
    parser.add_argument(
        '--imgsz', '-img',
        type=int,
        default=640,
        help='输入图片尺寸'
    )
    
    # 模型配置
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/weights/best.pt',
        help='预训练模型路径 (默认使用项目内best.pt)'
    )
    parser.add_argument(
        '--device', '-dev',
        type=str,
        default='0',
        help='训练设备 (GPU编号，如 0/1 或 cpu)'
    )
    
    # 输出配置
    parser.add_argument(
        '--project', '-p',
        type=str,
        default='models/train_results',
        help='项目保存目录（相对于项目根目录）'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default='exp',
        help='实验名称'
    )
    
    # 优化器配置
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        help='优化器类型 (SGD/Adam/AdamW/auto)'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='初始学习率'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='最终学习率因子'
    )
    parser.add_argument(
        '--cos_lr',
        action='store_true',
        help='使用余弦学习率'
    )
    
    # 训练策略
    parser.add_argument(
        '--patience', '-pat',
        type=int,
        default=50,
        help='早停耐心值'
    )
    parser.add_argument(
        '--save_period',
        type=int,
        default=10,
        help='模型保存间隔'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='使用自动混合精度训练'
    )
    parser.add_argument(
        '--close_mosaic',
        type=int,
        default=10,
        help='最后N轮关闭mosaic增强'
    )
    
    # 恢复训练
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='恢复训练'
    )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 确保从项目目录运行
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    args = parse_args()

    # 自动检测可用设备，如果GPU不可用则自动切换到CPU
    device = get_available_device(args.device)

    # 开始训练
    results = train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model=args.model,
        device=device,
        project=args.project,
        name=args.name,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        patience=args.patience,
        save_period=args.save_period,
        amp=args.amp,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
    )
    
    return results


if __name__ == '__main__':
    main()
