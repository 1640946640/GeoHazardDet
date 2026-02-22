# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  数据转换模块 - Shp转YOLO格式
# ============================================================
#
# 【模块功能】
# - 将Shapefile地理空间数据转换为YOLO格式标注
# - 自动划分训练集/验证集/测试集
# - 支持自定义类别映射
#
# 【注意事项】
# 1. 本脚本处理的是Shapefile格式的矢量数据
# 2. YOLO格式需要配合图片数据使用
# 3. 转换前需确保数据完整性
#
# 【使用示例】
# ```python
# from src.data.converter import ShpToYoloConverter
#
# # 初始化转换器
# converter = ShpToYoloConverter(
#     shp_dir='datasets/全国地质灾害shp/',
#     output_dir='datasets/',
#     class_mapping={0: '泥石流', 1: '滑坡'}
# )
#
# # 执行转换
# converter.convert(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
# ```
#
# 【数据格式说明】
# - 输入: Shapefile (.shp) - GIS矢量数据格式
# - 输出: YOLO格式标注文件 (.txt)
# - 标签结构: class_id x_center y_center width height
#
# ============================================================

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import shapefile
except ImportError:
    shapefile = None


class ShpToYoloConverter:
    """
    Shapefile转YOLO格式转换器
    
    用于将地理信息系统（GIS）中的Shapefile矢量数据
    转换为YOLO目标检测格式的标注文件
    
    【Shapefile格式说明】
    - .shp: 存储几何要素（点、线、面）
    - .dbf: 存储属性数据（要素特征）
    - .prj: 存储坐标投影信息
    
    【YOLO格式说明】
    - 每行一个目标: <class_id> <x_center> <y_center> <width> <height>
    - 坐标均为归一化值（0-1范围）
    - class_id从0开始编号
    """
    
    # 地质灾害类别映射
    # 默认: 0=泥石流, 1=滑坡
    DEFAULT_CLASS_MAPPING = {
        0: 'debris_flow',      # 泥石流
        1: 'landslide',         # 滑坡
    }
    
    def __init__(
        self,
        shp_dir: str = 'datasets/全国地质灾害shp/',
        output_dir: str = 'datasets/',
        class_mapping: Optional[Dict[int, str]] = None,
        random_seed: int = 42,
    ):
        """
        初始化转换器
        
        Args:
            shp_dir: Shapefile目录路径
            output_dir: 输出目录路径
            class_mapping: 类别映射字典 {类别ID: 类别名称}
            random_seed: 随机种子，保证数据划分可复现
        """
        self.shp_dir = Path(shp_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping or self.DEFAULT_CLASS_MAPPING
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(self.random_seed)
        
    def get_shapefiles(self) -> List[Path]:
        """
        获取目录下所有Shapefile文件
        
        Returns:
            Shapefile文件路径列表
        """
        if not self.shp_dir.exists():
            raise FileNotFoundError(f"Shapefile目录不存在: {self.shp_dir}")
        
        shp_files = list(self.shp_dir.glob('*.shp'))
        
        if not shp_files:
            print(f"警告: 在 {self.shp_dir} 中未找到.shp文件")
            
        return shp_files
    
    def read_shapefile(self, shp_path: Path) -> Tuple[List, Dict]:
        """
        读取Shapefile文件内容
        
        Args:
            shp_path: Shapefile路径
            
        Returns:
            几何数据列表和属性数据字典
        """
        if shapefile is None:
            raise ImportError(
                "pyshp库未安装，请先安装: pip install pyshp"
            )
        
        try:
            # 读取Shapefile
            with shapefile.Reader(str(shp_path)) as sf:
                # 获取所有记录
                records = sf.records()
                # 获取几何形状
                shapes = sf.shapes()
                
                return list(zip(records, shapes))
        except Exception as e:
            print(f"读取Shapefile失败 {shp_path}: {e}")
            return []
    
    def extract_yolo_labels(
        self,
        records_and_shapes: List[Tuple]
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        从Shapefile数据中提取YOLO格式标签
        
        Args:
            records_and_shapes: (记录, 形状)元组列表
            
        Returns:
            YOLO格式标签列表 [(class_id, x_center, y_center, width, height), ...]
        """
        labels = []
        
        for record, shape in records_and_shapes:
            # 根据属性数据判断灾害类型
            # 假设Shapefile中有类型字段
            disaster_type = self._identify_disaster_type(record)
            
            if disaster_type is None:
                continue
            
            # 转换几何数据为YOLO格式边界框
            bbox = self._shape_to_bbox(shape)
            
            if bbox:
                class_id = self._get_class_id(disaster_type)
                labels.append((class_id,) + bbox)
        
        return labels
    
    def _identify_disaster_type(self, record) -> Optional[str]:
        """
        识别灾害类型
        
        Args:
            record: Shapefile记录
            
        Returns:
            灾害类型字符串或None
        """
        # TODO: 根据实际Shapefile的字段名调整
        # 常见字段名示例: 'type', 'disaster_type', 'TYPE', '类型'
        
        # 尝试多种可能的字段名
        possible_fields = ['type', 'disaster_type', 'TYPE', '类型', '灾害类型']
        
        for field in possible_fields:
            try:
                if hasattr(record, field):
                    value = getattr(record, field)
                elif isinstance(record, dict) and field in record:
                    value = record[field]
                else:
                    # 尝试按索引访问
                    value = record[0]
                    
                if isinstance(value, str):
                    value = value.lower()
                    if '泥石流' in value or 'debris' in value:
                        return 'debris_flow'
                    elif '滑坡' in value or 'landslide' in value:
                        return 'landslide'
            except (AttributeError, KeyError, IndexError):
                continue
        
        return None
    
    def _shape_to_bbox(
        self,
        shape
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        将Shapefile几何形状转换为边界框
        
        Args:
            shape: Shapefile几何形状
            
        Returns:
            (x_center, y_center, width, height) 归一化坐标
        """
        # 获取形状的边界框
        bbox = shape.bbox  # [xmin, ymin, xmax, ymax]
        
        if not bbox or len(bbox) != 4:
            return None
        
        xmin, ymin, xmax, ymax = bbox
        
        # 计算中心点和宽高
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        
        # 归一化（假设坐标系范围为0-1，实际需根据数据范围调整）
        # 如果是经纬度坐标，需要根据实际范围归一化
        # 这里提供基础实现，实际使用需根据数据调整
        
        return (x_center, y_center, width, height)
    
    def _get_class_id(self, disaster_type: str) -> int:
        """
        获取类别ID
        
        Args:
            disaster_type: 灾害类型
            
        Returns:
            类别ID
        """
        # 根据灾害类型返回对应的类别ID
        type_to_id = {
            'debris_flow': 0,
            'landslide': 1,
        }
        
        return type_to_id.get(disaster_type, 0)
    
    def create_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test',
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return dirs
    
    def split_dataset(
        self,
        labels: List,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> Tuple[List, List, List]:
        """
        划分训练集/验证集/测试集
        
        Args:
            labels: 标签列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        # 打乱数据
        shuffled_labels = labels.copy()
        random.shuffle(shuffled_labels)
        
        total = len(shuffled_labels)
        
        # 计算分割点
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # 分割数据集
        train_set = shuffled_labels[:train_end]
        val_set = shuffled_labels[train_end:val_end]
        test_set = shuffled_labels[val_end:]
        
        return train_set, val_set, test_set
    
    def save_labels(
        self,
        labels: List,
        set_name: str,
    ):
        """
        保存标签文件
        
        Args:
            labels: 标签列表
            set_name: 数据集划分名称（train/val/test）
        """
        label_dir = self.output_dir / 'labels' / set_name
        
        for idx, label_data in enumerate(labels):
            # 生成文件名（使用索引作为文件名）
            filename = f"{idx:06d}.txt"
            filepath = label_dir / filename
            
            # 写入YOLO格式标签
            class_id, x_center, y_center, width, height = label_data
            
            with open(filepath, 'w', encoding='utf-8') as f:
                # YOLO格式: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def convert(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        min_samples: int = 10,
    ) -> Dict:
        """
        执行数据转换
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            min_samples: 每个类别最小样本数
            
        Returns:
            转换统计信息
        """
        print("=" * 60)
        print("Shapefile转YOLO格式转换器")
        print("=" * 60)
        
        # 1. 创建输出目录
        print("\n[1/5] 创建输出目录...")
        output_dirs = self.create_output_dirs()
        for dir_path in output_dirs:
            print(f"  - {dir_path}")
        
        # 2. 获取Shapefile列表
        print("\n[2/5] 扫描Shapefile文件...")
        shp_files = self.get_shapefiles()
        print(f"  找到 {len(shp_files)} 个Shapefile文件")
        
        # 3. 读取并转换数据
        print("\n[3/5] 读取并转换数据...")
        all_labels = []
        
        for shp_path in shp_files:
            print(f"  处理: {shp_path.name}")
            
            records_and_shapes = self.read_shapefile(shp_path)
            labels = self.extract_yolo_labels(records_and_shapes)
            
            print(f"    - 提取到 {len(labels)} 个标注")
            all_labels.extend(labels)
        
        if not all_labels:
            print("\n警告: 未提取到有效标签，请检查Shapefile数据格式")
            print("提示: 确保Shapefile中包含灾害类型字段（如'type'或'类型'）")
            
        # 4. 划分数据集
        print("\n[4/5] 划分数据集...")
        train_set, val_set, test_set = self.split_dataset(
            all_labels,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        
        print(f"  - 训练集: {len(train_set)} 样本")
        print(f"  - 验证集: {len(val_set)} 样本")
        print(f"  - 测试集: {len(test_set)} 样本")
        
        # 5. 保存标签文件
        print("\n[5/5] 保存标签文件...")
        self.save_labels(train_set, 'train')
        self.save_labels(val_set, 'val')
        self.save_labels(test_set, 'test')
        
        # 统计信息
        stats = {
            'total_samples': len(all_labels),
            'train_samples': len(train_set),
            'val_samples': len(val_set),
            'test_samples': len(test_set),
            'num_classes': len(self.class_mapping),
            'class_mapping': self.class_mapping,
        }
        
        print("\n" + "=" * 60)
        print("转换完成！统计信息：")
        print(f"  - 总样本数: {stats['total_samples']}")
        print(f"  - 训练集: {stats['train_samples']} ({train_ratio*100:.0f}%)")
        print(f"  - 验证集: {stats['val_samples']} ({val_ratio*100:.0f}%)")
        print(f"  - 测试集: {stats['test_samples']} ({test_ratio*100:.0f}%)")
        print(f"  - 类别数: {stats['num_classes']}")
        for class_id, class_name in self.class_mapping.items():
            print(f"    - ID {class_id}: {class_name}")
        print("=" * 60)
        
        return stats


def main():
    """
    主函数 - 命令行入口
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Shapefile转YOLO格式转换器'
    )
    parser.add_argument(
        '--shp-dir',
        type=str,
        default='datasets/全国地质灾害shp/',
        help='Shapefile目录路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='datasets/',
        help='输出目录路径'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='训练集比例 (默认: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='验证集比例 (默认: 0.2)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例 (默认: 0.1)'
    )
    
    args = parser.parse_args()
    
    # 创建转换器并执行转换
    converter = ShpToYoloConverter(
        shp_dir=args.shp_dir,
        output_dir=args.output_dir,
    )
    
    stats = converter.convert(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    return stats


if __name__ == '__main__':
    main()
