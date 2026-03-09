# ============================================================
#  泥石流/滑坡地质灾害识别项目
#  日志记录模块
# ============================================================
#
# 【功能】
# 提供训练过程日志记录，支持控制台和文件输出
#
# ============================================================
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


class Logger:
    """日志记录器类.

    提供完整的日志记录功能，包括：
    - 多级别日志（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    - 控制台和文件双输出
    - 训练指标记录

    【使用示例】
    ```python
    logger = Logger(log_dir="logs/", level=logging.INFO)
    logger.info("训练开始")
    logger.log_metrics({"loss": 0.5}, epoch=1)
    ```
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
            log_dir: 日志保存目录
            log_file: 日志文件名（None则自动生成）
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

        # 创建logger
        self.logger = logging.getLogger("GeoHazardDet")
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # 文件处理器
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(level)
        formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # 控制台处理器
        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def debug(self, msg: str):
        """调试信息."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """一般信息."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """警告信息."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """错误信息."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """严重错误."""
        self.logger.critical(msg)

    def log_metrics(self, metrics: dict[str, float], epoch: int, phase: str = "train"):
        """记录训练指标.

        Args:
            metrics: 指标字典 {'metric_name': value}
            epoch: 当前轮次
            phase: 训练阶段
        """
        self.info(f"\n[{phase.upper()}] Epoch {epoch}")
        for name, value in metrics.items():
            self.info(f"  {name}: {value:.4f}")


# 创建全局默认logger实例
_default_logger: Logger | None = None


def get_logger(log_dir: str = "logs/") -> Logger:
    """获取全局Logger实例（单例模式）.

    Args:
        log_dir: 日志目录

    Returns:
        Logger实例
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger(log_dir=log_dir)
    return _default_logger
