import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Union, Tuple
import json
from pathlib import Path
import time
import warnings
from functools import partial
import inspect

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DataProcessor')

# 禁用不必要的警告
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataProcessor:
    """
    高效数据处理器，具有以下优化特性：
    1. 增强的异常处理
    2. 并行处理支持
    3. 内存优化
    4. 动态指标添加
    5. 详细日志记录
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        max_workers: int = 4,
        memory_limit: Optional[int] = None,
        log_level: str = 'INFO'
    ):
        """
        初始化数据处理器
        
        Args:
            chunk_size: 分块处理的大小
            max_workers: 最大并行工作数
            memory_limit: 内存限制(MB)，None表示无限制
            log_level: 日志级别
        """
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.memory_limit = memory_limit
        self.metrics: Dict[str, Callable] = {}
        self._processing_stats: Dict[str, Union[float, Dict]] = {}
        self._setup_logging(log_level)

    def _setup_logging(self, level: str) -> None:
        """配置日志级别"""
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.info("DataProcessor initialized with chunk_size=%d, max_workers=%d", 
                   self.chunk_size, self.max_workers)

    def add_metric(
        self, 
        name: str, 
        metric_func: Callable,
        params: Optional[Dict] = None,
        overwrite: bool = False
    ) -> None:
        """
        动态添加自定义指标
        
        Args:
            name: 指标名称
            metric_func: 指标计算函数
            params: 指标参数
            overwrite: 是否覆盖已有指标
            
        Raises:
            ValueError: 如果指标已存在且overwrite为False
        """
        if name in self.metrics and not overwrite:
            raise ValueError(f"指标 '{name}' 已存在，设置 overwrite=True 来覆盖")
        
        if params:
            # 使用偏函数固定参数
            self.metrics[name] = partial(metric_func, **params)
            logger.debug("添加指标 '%s' 带参数: %s", name, params)
        else:
            self.metrics[name] = metric_func
            logger.debug("添加指标 '%s'", name)

    def process_data(
        self,
        df: pd.DataFrame,
        metrics_to_apply: Optional[List[str]] = None,
        output_format: str = 'pandas'
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        处理数据，应用指定的指标计算
        
        Args:
            df: 输入数据框
            metrics_to_apply: 要应用的指标列表，None表示应用所有
            output_format: 输出格式 ('pandas' 或 'dict')
            
        Returns:
            处理后的数据
            
        Raises:
            MemoryError: 如果超过内存限制
            ValueError: 如果输入数据为空或指标无效
        """
        start_time = time.time()
        self._validate_input(df, metrics_to_apply)
        
        try:
            # 内存检查
            self._check_memory_usage(df)
            
            # 确定要应用的指标
            metrics = self._resolve_metrics(metrics_to_apply)
            logger.info("开始处理数据，应用指标: %s", ', '.join(metrics.keys()))
            
            # 分块处理
            if len(df) > self.chunk_size and self.max_workers > 1:
                logger.info("启用并行处理 (%d workers)", self.max_workers)
                result = self._process_in_parallel(df, metrics)
            else:
                logger.info("使用单线程处理")
                result = self._process_chunk(df, metrics)
            
            # 转换输出格式
            processed_data = self._format_output(result, output_format)
            
            # 记录处理统计信息
            self._processing_stats = {
                'total_time': time.time() - start_time,
                'metrics_applied': list(metrics.keys()),
                'input_shape': df.shape,
                'output_shape': processed_data.shape if hasattr(processed_data, 'shape') else None
            }
            
            logger.info("数据处理完成，耗时 %.2f 秒", self._processing_stats['total_time'])
            return processed_data
            
        except Exception as e:
            logger.error("数据处理失败: %s", str(e), exc_info=True)
            raise

    def _validate_input(
        self, 
        df: pd.DataFrame, 
        metrics_to_apply: Optional[List[str]]
    ) -> None:
        """验证输入数据和指标"""
        if df.empty:
            logger.error("输入数据为空")
            raise ValueError("输入数据不能为空")
            
        if metrics_to_apply is None:
            return
            
        missing_metrics = set(metrics_to_apply) - set(self.metrics.keys())
        if missing_metrics:
            logger.error("请求的指标不存在: %s", missing_metrics)
            raise ValueError(f"以下指标未定义: {missing_metrics}")

    def _check_memory_usage(self, df: pd.DataFrame) -> None:
        """检查内存使用情况"""
        if self.memory_limit is None:
            return
            
        mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        if mem_usage > self.memory_limit:
            logger.error("内存使用超过限制 (%.2f MB > %d MB)", mem_usage, self.memory_limit)
            raise MemoryError(f"内存使用超过限制 ({mem_usage:.2f} MB > {self.memory_limit} MB)")
        else:
            logger.debug("当前内存使用: %.2f MB / %s MB", 
                        mem_usage, 
                        str(self.memory_limit) if self.memory_limit else '无限制')

    def _resolve_metrics(
        self, 
        metrics_to_apply: Optional[List[str]]
    ) -> Dict[str, Callable]:
        """解析要应用的指标"""
        if metrics_to_apply is None:
            return self.metrics.copy()
        return {name: self.metrics[name] for name in metrics_to_apply}

    def _process_in_parallel(
        self, 
        df: pd.DataFrame, 
        metrics: Dict[str, Callable]
    ) -> Dict[str, np.ndarray]:
        """
        并行处理数据
        
        Args:
            df: 输入数据框
            metrics: 要应用的指标字典
            
        Returns:
            包含所有指标结果的字典
        """
        chunks = self._split_dataframe(df)
        total_chunks = len(chunks)
        logger.info("将数据分成 %d 个块进行并行处理", total_chunks)
        
        results = {}
        futures = {}
        
        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                for i, chunk in enumerate(chunks):
                    future = executor.submit(self._process_chunk, chunk, metrics)
                    futures[future] = i
                    logger.debug("已提交块 %d/%d 处理", i+1, total_chunks)
                
                # 收集结果
                for future in as_completed(futures):
                    chunk_idx = futures[future]
                    try:
                        chunk_result = future.result()
                        self._merge_results(results, chunk_result)
                        logger.debug("已完成块 %d/%d 处理", chunk_idx+1, total_chunks)
                    except Exception as e:
                        logger.error("处理块 %d 失败: %s", chunk_idx, str(e), exc_info=True)
                        raise
                        
        except Exception as e:
            logger.error("并行处理失败: %s", str(e), exc_info=True)
            raise
            
        return results

    def _split_dataframe(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """将数据框分割成块"""
        chunks = []
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _process_chunk(
        self, 
        chunk: pd.DataFrame, 
        metrics: Dict[str, Callable]
    ) -> Dict[str, np.ndarray]:
        """
        处理单个数据块
        
        Args:
            chunk: 数据块
            metrics: 要应用的指标字典
            
        Returns:
            包含该块指标结果的字典
        """
        chunk_results = {}
        
        for name, metric_func in metrics.items():
            try:
                start_time = time.time()
                
                # 应用指标函数
                result = metric_func(chunk)
                
                # 验证结果格式
                if not isinstance(result, (pd.DataFrame, pd.Series, np.ndarray)):
                    logger.warning("指标 '%s' 返回了不支持的类型: %s", name, type(result))
                    continue
                
                # 转换为numpy数组节省内存
                chunk_results[name] = result.values if hasattr(result, 'values') else result
                
                processing_time = time.time() - start_time
                logger.debug("应用指标 '%s' 耗时 %.4f 秒", name, processing_time)
                
            except Exception as e:
                logger.error("应用指标 '%s' 失败: %s", name, str(e), exc_info=True)
                raise
                
        return chunk_results

    def _merge_results(
        self, 
        main_results: Dict[str, np.ndarray], 
        chunk_results: Dict[str, np.ndarray]
    ) -> None:
        """合并分块处理结果"""
        for name, result in chunk_results.items():
            if name in main_results:
                main_results[name] = np.concatenate((main_results[name], result))
            else:
                main_results[name] = result

    def _format_output(
        self, 
        result: Dict[str, np.ndarray], 
        output_format: str
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """格式化输出"""
        if output_format == 'pandas':
            return pd.DataFrame(result)
        elif output_format == 'dict':
            return result
        else:
            logger.warning("未知的输出格式 '%s'，默认使用 pandas", output_format)
            return pd.DataFrame(result)

    def get_processing_stats(self) -> Dict[str, Union[float, Dict]]:
        """获取处理统计信息"""
        return self._processing_stats.copy()

    @staticmethod
    def example_metric(df: pd.DataFrame, window: int = 5) -> pd.Series:
        """示例指标：简单移动平均"""
        return df['close'].rolling(window=window).mean()

    def mark_suspend(self, df):
        """标记停牌日（全NaN为停牌）"""
        df['is_suspend'] = df.isnull().all(axis=1).astype(int)
        return df

# 示例用法
if __name__ == "__main__":
    try:
        # 创建处理器实例
        processor = DataProcessor(
            chunk_size=5000,
            max_workers=4,
            memory_limit=1024,  # 1GB
            log_level='DEBUG'
        )
        
        # 添加示例指标
        processor.add_metric('sma5', DataProcessor.example_metric, {'window': 5})
        processor.add_metric('sma10', DataProcessor.example_metric, {'window': 10})
        
        # 生成测试数据
        np.random.seed(42)
        test_data = pd.DataFrame({
            'close': np.cumsum(np.random.randn(10000)) + 100,
            'volume': np.random.randint(100, 10000, size=10000)
        })
        
        # 处理数据
        result = processor.process_data(
            test_data,
            metrics_to_apply=['sma5', 'sma10'],
            output_format='pandas'
        )
        
        # 打印结果和统计信息
        print("\n处理结果前5行:")
        print(result.head())
        
        print("\n处理统计信息:")
        print(processor.get_processing_stats())
        
    except Exception as e:
        logger.critical("示例运行失败: %s", str(e), exc_info=True)