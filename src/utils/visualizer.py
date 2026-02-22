# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  可视化模块
# ============================================================
#
# 【功能】
# 提供训练曲线、检测结果、混淆矩阵等可视化功能
#
# 【主要类】
# - Visualizer: 结果可视化工具
#
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Dict, List, Optional


class Visualizer:
    """
    结果可视化工具
    
    提供训练曲线、检测结果等可视化功能
    
    【功能】
    - 绘制训练曲线
    - 可视化检测结果
    - 绘制混淆矩阵
    - 绘制类别分布
    
    """
    
    # 类别颜色配置 (BGR格式，用于绘制)
    COLORS = {
        0: (255, 0, 0),      # 泥石流 - 红色
        1: (0, 255, 0),      # 滑坡 - 绿色
    }
    
    # 类别名称
    CLASS_NAMES = {
        0: '泥石流',
        1: '滑坡',
    }
    
    def __init__(
        self,
        save_dir: str = 'runs/visualize/',
        dpi: int = 150,
    ):
        """
        初始化可视化工具
        
        Args:
            save_dir: 保存目录
            dpi: 图片分辨率
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
    def plot_training_curves(
        self,
        epochs: List[int],
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None,
        title: str = 'Training Curves',
    ):
        """
        绘制训练曲线
        
        Args:
            epochs: 轮次列表
            metrics: 指标字典 {指标名: [值列表]}
            save_path: 保存路径
            title: 图表标题
        """
        plt.figure(figsize=(12, 8))
        
        for metric_name, values in metrics.items():
            plt.plot(epochs, values, label=metric_name, linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        
        plt.close()
    
    def visualize_detection(
        self,
        image: np.ndarray,
        detections: np.ndarray,
        class_names: Optional[Dict[int, str]] = None,
        conf_threshold: float = 0.5,
        save_path: Optional[str] = None,
    ):
        """
        可视化检测结果
        
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
        image_copy = image.copy().astype(np.uint8)
        
        # 转换为PIL Image
        pil_image = Image.fromarray(image_copy)
        draw = ImageDraw.Draw(pil_image)
        
        # 获取图片尺寸
        img_width, img_height = pil_image.size
        
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
            class_name = class_names.get(cls_id, f'Class_{cls_id}')
            
            # 绘制边界框
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=3
            )
            
            # 绘制标签
            label = f'{class_name}: {conf:.2f}'
            
            # 标签背景
            try:
                text_width, text_height = draw.textsize(label)
            except AttributeError:
                # PIL版本兼容
                left, top, right, bottom = draw.textbbox((0, 0), label)
                text_width, text_height = right - left, bottom - top
            
            draw.rectangle(
                [x1, y1 - text_height - 5, x1 + text_width + 5, y1],
                fill=color
            )
            
            # 标签文字
            draw.text(
                (x1 + 3, y1 - text_height - 3),
                label,
                fill=(255, 255, 255)
            )
        
        # 保存结果
        if save_path:
            pil_image.save(save_path)
            print(f"检测结果已保存: {save_path}")
        
        return np.array(pil_image)
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        绘制混淆矩阵
        
        Args:
            confusion_matrix: 混淆矩阵
            class_names: 类别名称列表
            save_path: 保存路径
            normalize: 是否归一化
        """
        if class_names is None:
            class_names = ['debris_flow', 'landslide']
        
        if normalize:
            cm = confusion_matrix.astype('float') / (
                confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10
            )
        else:
            cm = confusion_matrix
        
        plt.figure(figsize=(8, 6))
        
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # 在格子中添加数值
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, f'{cm[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black'
                )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")
        
        plt.close()
    
    def plot_class_distribution(
        self,
        class_counts: Dict[str, int],
        save_path: Optional[str] = None,
    ):
        """
        绘制类别分布图
        
        Args:
            class_counts: 类别计数字典
            save_path: 保存路径
        """
        plt.figure(figsize=(8, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        colors = ['#FF6B6B', '#4ECDC4']
        
        plt.bar(classes, counts, color=colors, edgecolor='black', linewidth=1.5)
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Class Distribution', fontsize=14)
        
        for i, (cls, cnt) in enumerate(zip(classes, counts)):
            plt.text(i, cnt + max(counts) * 0.02, str(cnt), 
                    ha='center', fontsize=12, fontweight='bold')
        
        plt.ylim(0, max(counts) * 1.15)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"类别分布图已保存: {save_path}")
        
        plt.close()


# 创建默认可视化器实例
_default_visualizer: Optional[Visualizer] = None


def get_visualizer(save_dir: str = 'runs/visualize/') -> Visualizer:
    """
    获取全局Visualizer实例（单例模式）
    
    Args:
        save_dir: 保存目录
        
    Returns:
        Visualizer实例
    """
    global _default_visualizer
    if _default_visualizer is None:
        _default_visualizer = Visualizer(save_dir=save_dir)
    return _default_visualizer
