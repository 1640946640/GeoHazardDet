# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  推理脚本 - 图片/视频/批量检测
# ============================================================
#
# 【基于Ultralytics YOLOv8开发】
# 本项目基于Ultralytics YOLOv8框架开发，遵循AGPL-3.0许可证
# 参考: https://github.com/ultralytics/ultralytics
#
# 【功能】
# - 单张图片推理
# - 批量图片推理
# - 视频流推理
# - 实时摄像头推理
# - 结果可视化与保存
#
# 【使用方法】
# python src/predict.py --source test.jpg  # 使用项目内best.pt检测单张图片
#
# # 批量图片检测
# python src/predict.py --source images/
#
# # 视频检测
# python src/predict.py --source video.mp4
#
# # 实时摄像头
# python src/predict.py --source 0
#
# ============================================================

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image


def get_available_device(requested_device: str = '0') -> str:
    """
    自动检测可用的推理设备

    如果请求的设备不可用（ 如 NVIDIA GPU），自动回退到 CPU

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


class DisasterPredictor:
    """
    地质灾害检测推理器
    
    支持图片、视频、批量检测的推理类
    
    【功能】
    - 单张/批量图片检测
    - 视频文件检测
    - 实时摄像头检测
    - 检测结果可视化
    - 结果保存
    """
    
    # 类别名称映射
    CLASS_NAMES = {
        0: '泥石流',
        1: '滑坡',
    }
    
    # 类别颜色 (BGR格式)
    CLASS_COLORS = {
        0: (0, 0, 255),      # 红色 - 泥石流
        1: (0, 255, 0),      # 绿色 - 滑坡
    }
    
    def __init__(
        self,
        weights: str,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        imgsz: int = 640,
        device: str = '0',
        half: bool = False,
        classes: Optional[List[int]] = None,
        save_dir: str = 'models/predict_results',  # 项目内相对路径
    ):
        """
        初始化推理器

        Args:
            weights: 模型权重文件路径
            conf_thres: 置信度阈值
            iou_thres: NMS的IoU阈值
            imgsz: 输入图片尺寸
            classes: 只检测指定类别
            save_dir: 结果保存目录（项目内相对路径）
        """
        self.weights = Path(weights)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        # 自动检测可用设备，如果GPU不可用则自动切换到CPU
        self.device = get_available_device(device)
        self.half = half
        self.classes = classes
        self.save_dir = Path(save_dir)
        
        # 加载模型
        self.model = self._load_model()
        
    def _load_model(self):
        """
        加载YOLOv8模型
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("错误: 未安装ultralytics库")
            print("请运行: pip install ultralytics>=8.1.0")
            sys.exit(1)
        
        print(f"加载模型: {self.weights}")
        model = YOLO(str(self.weights))
        return model
    
    def predict(
        self,
        source: Union[str, np.ndarray, Image.Image],
        save: bool = True,
        save_dir: str = 'models/predict_results',  # 项目内相对路径
        name: str = 'exp',
        draw_bbox: bool = True,
        show_conf: bool = True,
    ) -> List[Dict]:
        """
        执行推理
        
        Args:
            source: 输入源 (图片路径/图片数组/PIL Image/目录)
            save: 是否保存结果
            save_dir: 结果保存目录
            name: 子目录名称
            draw_bbox: 是否绘制边界框
            show_conf: 是否显示置信度
            name: 实验名称
            
        Returns:
            检测结果列表
        """
        # 确保保存目录是绝对路径（项目内）
        import os
        if not os.path.isabs(save_dir):
            save_dir = str(Path.cwd() / save_dir)
        
        results = self.model.predict(
            source=source,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            classes=self.classes,
            save=save,
            project=save_dir,
            name=name,
            exist_ok=True,
        )
        
        # 处理结果
        all_results = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    pred = {
                        'class_id': int(box.cls[0]),
                        'class_name': self.CLASS_NAMES.get(int(box.cls[0]), 'Unknown'),
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    }
                    all_results.append(pred)
                    
                    # 绘制边界框
                    if draw_bbox:
                        self._draw_bbox(result.orig_img, box)
        
        return all_results
    
    def _draw_bbox(self, image: np.ndarray, box):
        """
        在图片上绘制边界框
        
        Args:
            image: 图片数组 (BGR格式)
            box: 检测框
        """
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # 获取颜色
        color = self.CLASS_COLORS.get(cls_id, (0, 255, 255))
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        
        # 绘制标签
        label = f"{self.CLASS_NAMES.get(cls_id, 'Unknown')}: {conf:.2f}"
        
        # 计算标签背景
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # 标签背景矩形
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # 标签文字
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    def predict_single_image(
        self,
        image_path: str,
        save_result: bool = True,
        output_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        单张图片推理
        
        Args:
            image_path: 图片路径
            save_result: 是否保存结果
            output_path: 结果保存路径
            
        Returns:
            检测结果列表
        """
        print(f"\n检测图片: {image_path}")
        
        results = self.predict(
            source=image_path,
            save=save_result,
            save_dir=str(self.save_dir),
            name='single',
        )
        
        print(f"检测到 {len(results)} 个目标:")
        for i, pred in enumerate(results):
            print(f"  [{i+1}] {pred['class_name']}: 置信度={pred['confidence']:.4f}")
            print(f"      位置: {pred['bbox']}")
        
        return results
    
    def predict_batch(
        self,
        image_dir: str,
        save_results: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        批量图片推理
        
        Args:
            image_dir: 图片目录路径
            save_results: 是否保存结果
            
        Returns:
            检测结果字典 {图片路径: 检测结果列表}
        """
        print(f"\n批量检测目录: {image_dir}")
        
        # 获取所有图片
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff']
        image_files = []
        for ext in supported_formats:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        image_files = sorted(image_files)
        print(f"找到 {len(image_files)} 张图片")
        
        if not image_files:
            print("错误: 未找到图片文件")
            return {}
        
        # 批量推理
        all_results = {}
        for img_path in image_files:
            results = self.predict_single_image(
                str(img_path),
                save_result=save_results,
            )
            all_results[str(img_path)] = results
        
        # 统计
        total_detections = sum(len(r) for r in all_results.values())
        print(f"\n批量检测完成！")
        print(f"  - 处理图片: {len(image_files)} 张")
        print(f"  - 检测目标: {total_detections} 个")
        
        return all_results
    
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_video: bool = True,
        fps: int = 30,
    ) -> None:
        """
        视频推理
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            save_video: 是否保存结果视频
            fps: 输出视频帧率
        """
        print(f"\n检测视频: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在: {video_path}")
            return
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("错误: 无法打开视频")
            return
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {original_fps}fps, {total_frames}帧")
        
        # 输出视频配置
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 帧计数器
        frame_idx = 0
        detections_count = 0
        
        # 进度条
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="处理帧")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_idx += 1
            
            # 推理
            results = self.predict(
                source=frame,
                save=False,
            )
            
            detections_count += len(results)
            
            # 在帧上绘制结果
            for pred in results:
                x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
                cls_id = pred['class_id']
                conf = pred['confidence']
                
                color = self.CLASS_COLORS.get(cls_id, (0, 255, 255))
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{pred['class_name']}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 保存帧
            if save_video and output_path:
                out.write(frame)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if save_video and output_path:
            out.release()
        
        print(f"\n视频处理完成！")
        print(f"  - 处理帧数: {frame_idx}")
        print(f"  - 检测目标: {detections_count}")
    
    def predict_camera(
        self,
        camera_id: int = 0,
        window_name: str = '地质灾害检测',
    ) -> None:
        """
        实时摄像头推理
        
        Args:
            camera_id: 摄像头ID (0为默认摄像头)
            window_name: 窗口名称
        """
        print(f"\n启动摄像头检测 (按 'q' 退出)")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        # 创建窗口
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("错误: 无法读取帧")
                break
            
            # 推理
            results = self.predict(
                source=frame,
                save=False,
            )
            
            # 绘制结果
            for pred in results:
                x1, y1, x2, y2 = [int(c) for c in pred['bbox']]
                cls_id = pred['class_id']
                conf = pred['confidence']
                
                color = self.CLASS_COLORS.get(cls_id, (0, 255, 255))
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{pred['class_name']}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 显示帧
            cv2.imshow(window_name, frame)
            
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    """
    解析命令行参数
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='泥石流/滑坡地质灾害检测推理'
    )
    
    # 模型配置
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default= r'models/weights/best.pt',
        help='模型权重文件路径 (默认使用项目内best.pt)'
    )
    
    # 输入源
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='datasets/images/test/moxizheng_0.2m_UAV0095.jpg',
        help='输入源 (图片路径/视频路径/摄像头ID/目录，支持jpg/jpeg/png/tif/bmp/webp)'
    )
    
    # 推理参数
    parser.add_argument(
        '--conf-thres', '-conf',
        type=float,
        default=0.25,
        help='置信度阈值'
    )
    parser.add_argument(
        '--iou-thres', '-iou',
        type=float,
        default=0.45,
        help='NMS的IoU阈值'
    )
    parser.add_argument(
        '--imgsz', '-img',
        type=int,
        default=640,
        help='输入图片尺寸'
    )
    parser.add_argument(
        '--device', '-dev',
        type=str,
        default='0',
        help='推理设备 (GPU编号或cpu)'
    )
    
    # 输出配置
    parser.add_argument(
        '--save-dir',
        type=str,
        default='models/predict_results',
        help='结果保存目录 (项目内相对路径)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件路径 (视频/图片)'
    )
    
    # 推理模式
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='auto',
        help='推理模式 (image/video/camera/batch/auto)'
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
    
    # 检查权重文件
    if not os.path.exists(args.weights):
        print(f"错误: 模型文件不存在: {args.weights}")
        sys.exit(1)
    
    # 初始化推理器
    predictor = DisasterPredictor(
        weights=args.weights,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=args.imgsz,
        device=args.device,
    )
    
    # 根据模式执行推理
    source_path = Path(args.source)
    
    if args.mode == 'image' or (args.mode == 'auto' and source_path.is_file() and
                                 source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']):
        # 单张图片
        predictor.predict_single_image(
            args.source,
            save_result=True,
            output_path=args.output,
        )
        
    elif args.mode == 'batch' or (args.mode == 'auto' and source_path.is_dir()):
        # 批量图片
        predictor.predict_batch(args.source, save_results=True)
        
    elif args.mode == 'video' or (args.mode == 'auto' and 
                                   source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']):
        # 视频
        predictor.predict_video(
            args.source,
            output_path=args.output or 'runs/predict/result.mp4',
            save_video=True,
        )
        
    elif args.mode == 'camera' or (args.source.isdigit()):
        # 摄像头
        camera_id = int(args.source)
        predictor.predict_camera(camera_id=camera_id)
        
    else:
        # 默认：自动检测
        print(f"无法识别的输入类型: {args.source}")
        print("请使用 --mode 参数指定模式")


if __name__ == '__main__':
    main()
