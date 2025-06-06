"""
数据获取模块 - 使用Alpha Vantage API获取股票数据
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import sys
import os
import akshare as ak
import tushare as ts

# 配置API密钥
TS_TOKEN = "dfb371512cbe14cc65084a2dbdc5429990f605aa802d48bd2dd9146c"

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置Tushare Token
ts.set_token(TS_TOKEN)

class DataFetcher:
    def __init__(self):
        pass

    def get_akshare_data(self, symbol, market, freq='daily', adj='qfq', 
                     start_date=None, end_date=None):
        import akshare as ak
        import pandas as pd
        try:
            if market == 'A':
                df = ak.stock_zh_a_hist(symbol=symbol, 
                                      period=freq,
                                      start_date=start_date,
                                      end_date=end_date,
                                      adjust=adj)
                if df is None or df.empty:
                    return None
                # 获取A股交易日历
                trade_days = ak.tool_trade_date_hist_sina()
                trade_days = pd.to_datetime(trade_days['trade_date'])
                mask = (trade_days >= pd.to_datetime(start_date)) & (trade_days <= pd.to_datetime(end_date))
                all_days = trade_days[mask]
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').reindex(all_days)
                # 标记停牌日
                df['is_suspended'] = df['open'].isna().astype(int)
                df = df.reset_index().rename(columns={'index': '日期'})
            elif market == 'HK':
                df = ak.stock_hk_daily(symbol=symbol)
                if df is None or df.empty:
                    return None
                # 兼容不同列名
                if 'date' in df.columns:
                    df = df.rename(columns={'date': '日期'})
                if '日期' not in df.columns:
                    raise ValueError("港股数据缺少日期列")
                # 获取港股交易日历（用A股日历近似）
                trade_days = ak.tool_trade_date_hist_sina()
                trade_days = pd.to_datetime(trade_days['trade_date'])
                mask = (trade_days >= pd.to_datetime(start_date)) & (trade_days <= pd.to_datetime(end_date))
                all_days = trade_days[mask]
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').reindex(all_days)
                df['is_suspended'] = df['open'].isna().astype(int)
                df = df.reset_index().rename(columns={'index': '日期'})
            elif market == 'US':
                df = ak.stock_us_daily(symbol=symbol)
                if df is None or df.empty:
                    return None
                if 'date' in df.columns:
                    df = df.rename(columns={'date': '日期'})
                if '日期' not in df.columns:
                    raise ValueError("美股数据缺少日期列")
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
            else:
                raise ValueError(f"不支持的市场类型: {market}")
            return df
        except Exception as e:
            print(f"[WARN] akshare失败: {str(e)}")
            return None

    def get_tushare_data(self, symbol, freq, adj, start_date, end_date):
        # ... existing code ...
        pass

    def get_alpha_data(self, symbol, freq, interval):
        # ... existing code ...
        pass

    def get_data_with_fallback(self, symbol, market, freq='daily', adj='qfq', start_date=None, end_date=None, interval='5min'):
        """
        按优先级自动切换数据源，直到成功。
        A股/港股优先：akshare -> tushare -> alpha
        美股优先：alpha -> akshare
        """
        tried = []
        if market in ['A', 'HK']:
            # 1. akshare
            try:
                df = self.get_akshare_data(symbol, market, freq, adj, start_date, end_date)
                if df is not None and not df.empty:
                    print(f"[INFO] {market}优先: akshare成功")
                    return df
                tried.append('akshare')
            except Exception as e:
                print(f"[WARN] akshare失败: {e}")
                tried.append('akshare')
            # 2. tushare (仅A股)
            if market == 'A':
                try:
                    df = self.get_tushare_data(symbol, freq[0].upper(), adj, start_date, end_date)
                    if df is not None and not df.empty:
                        print("[INFO] tushare成功")
                        return df
                    tried.append('tushare')
                except Exception as e:
                    print(f"[WARN] tushare失败: {e}")
                    tried.append('tushare')
        elif market == 'US':
            # 1. akshare
            try:
                df = self.get_akshare_data(symbol, market, freq, adj, start_date, end_date)
                if df is not None and not df.empty:
                    print("[INFO] akshare成功")
                    return df
                tried.append('akshare')
            except Exception as e:
                print(f"[WARN] akshare失败: {e}")
                tried.append('akshare')
        else:
            raise ValueError('不支持的市场类型')
        raise RuntimeError(f"所有数据源均失败: {tried}")

    def _ensure_date_range(self, df, start_date, end_date):
        """
        补全指定日期区间的数据索引（以工作日或交易日为准），便于后续分析。
        """
        import pandas as pd
        # 判断日期列
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
        # 生成完整日期区间（可根据实际需求调整为交易日历）
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        df = df.reindex(date_range)
        df.index.name = '日期'
        return df.reset_index()

class DataConsistencyChecker:
    """数据一致性检查器"""
    
    def __init__(self):
        self.reference_fetcher = MultiMarketDataFetcher()
    
    def check_data_consistency(self, df, symbol, market, start_date, end_date):
        """
        检查数据一致性
        
        Parameters:
        df (pd.DataFrame): 待检查的数据
        symbol (str): 股票代码
        market (str): 市场类型 ('A', 'HK', 'US')
        start_date (str): 起始日期
        end_date (str): 结束日期
        
        Returns:
        dict: 检查结果报告
        """
        try:
            # 获取参考数据（使用akshare）
            ref_data = self.reference_fetcher.get_akshare_data(
                symbol=symbol,
                market=market,
                start_date=start_date,
                end_date=end_date
            )
            
            if ref_data is None or df is None:
                return {
                    'status': 'error',
                    'message': '无法获取参考数据或待检查数据为空',
                    'checks': {}
                }
            
            # 统一日期格式
            df.index = pd.to_datetime(df.index)
            ref_data.index = pd.to_datetime(ref_data.index)
            
            # 检查项目
            checks = {
                'date_range': {
                    'status': 'pass',
                    'details': {}
                },
                'data_completeness': {
                    'status': 'pass',
                    'details': {}
                },
                'value_consistency': {
                    'status': 'pass',
                    'details': {}
                }
            }
            
            # 1. 检查日期范围
            expected_start = pd.to_datetime(start_date)
            expected_end = pd.to_datetime(end_date)
            actual_start = df.index.min()
            actual_end = df.index.max()
            
            date_match = (
                abs((actual_start - expected_start).days) <= 1 and
                abs((actual_end - expected_end).days) <= 1
            )
            
            checks['date_range']['status'] = 'pass' if date_match else 'fail'
            checks['date_range']['details'] = {
                'expected_range': f"{expected_start.date()} to {expected_end.date()}",
                'actual_range': f"{actual_start.date()} to {actual_end.date()}"
            }
            
            # 2. 检查数据完整性
            expected_days = len(pd.date_range(start=start_date, end=end_date, freq='B'))
            actual_days = len(df)
            completeness_ratio = actual_days / expected_days
            
            checks['data_completeness']['status'] = 'pass' if completeness_ratio >= 0.95 else 'fail'
            checks['data_completeness']['details'] = {
                'expected_days': expected_days,
                'actual_days': actual_days,
                'completeness_ratio': f"{completeness_ratio:.2%}"
            }
            
            # 3. 检查数据一致性（与参考数据比较）
            common_dates = df.index.intersection(ref_data.index)
            if len(common_dates) > 0:
                # 计算收盘价差异
                close_diff_pct = abs(
                    (df.loc[common_dates, 'Close'] - ref_data.loc[common_dates, 'close']) /
                    ref_data.loc[common_dates, 'close']
                ).mean()
                
                checks['value_consistency']['status'] = 'pass' if close_diff_pct < 0.01 else 'fail'
                checks['value_consistency']['details'] = {
                    'avg_close_diff_pct': f"{close_diff_pct:.2%}",
                    'common_dates_count': len(common_dates)
                }
            
            # 汇总检查结果
            all_passed = all(check['status'] == 'pass' for check in checks.values())
            
            return {
                'status': 'pass' if all_passed else 'fail',
                'message': '所有检查通过' if all_passed else '存在不一致',
                'checks': checks
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'检查过程出错: {str(e)}',
                'checks': {}
            }
    
    def print_check_report(self, report):
        """打印检查报告"""
        print("\n=== 数据一致性检查报告 ===")
        print(f"状态: {'✅' if report['status'] == 'pass' else '❌'} {report['status']}")
        print(f"消息: {report['message']}")
        
        for check_name, check_result in report['checks'].items():
            print(f"\n{check_name}:")
            print(f"状态: {'✅' if check_result['status'] == 'pass' else '❌'} {check_result['status']}")
            for key, value in check_result['details'].items():
                print(f"  {key}: {value}")

# 在MultiMarketDataFetcher类中添加数据一致性检查方法
def check_data_consistency(self, df, symbol, market, start_date, end_date):
    """
    检查获取的数据是否满足一致性要求
    """
    checker = DataConsistencyChecker()
    return checker.check_data_consistency(df, symbol, market, start_date, end_date)

# 修改get_data_with_fallback方法，添加一致性检查
def get_data_with_fallback(self, symbol, market, freq='daily', adj='qfq', 
                          start_date=None, end_date=None, interval='5min',
                          check_consistency=True):
    """
    获取数据并进行一致性检查
    """
    df = super().get_data_with_fallback(symbol, market, freq, adj, start_date, end_date, interval)
    
    if check_consistency and df is not None:
        checker = DataConsistencyChecker()
        check_result = checker.check_data_consistency(df, symbol, market, start_date, end_date)
        
        if check_result['status'] != 'pass':
            print("\n⚠️ 数据一致性检查未通过:")
            checker.print_check_report(check_result)
            
    return df

class MultiMarketDataFetcher:
    def __init__(self):
        self.akshare_client = None  # 如果需要可以初始化 akshare
        self.tushare_client = None  # 如果需要可以初始化 tushare

    def _ensure_date_range(self, df, start_date, end_date):
        """
        补全指定日期区间的数据索引（以工作日或交易日为准），便于后续分析。
        """
        import pandas as pd
        # 判断日期列
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.set_index('日期')
        # 生成完整日期区间（可根据实际需求调整为交易日历）
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        df = df.reindex(date_range)
        df.index.name = '日期'
        return df.reset_index()

    def get_data_with_fallback(self, symbol, market, freq='daily', adj='qfq', 
                          start_date=None, end_date=None, interval='5min'):
        """获取市场数据，支持多个数据源自动切换"""
        tried = []
        
        # 验证日期参数
        if not start_date or not end_date:
            raise ValueError("start_date 和 end_date 是必需的参数")
            
        if market in ['A', 'HK']:
            # 1. akshare
            try:
                df = self.get_akshare_data(symbol, market, freq, adj, start_date, end_date)
                if df is not None and not df.empty:
                    print(f"[INFO] {market}优先: akshare成功")
                    return df
                tried.append('akshare')
            except Exception as e:
                print(f"[WARN] akshare失败: {e}")
                tried.append('akshare')
            # 2. tushare (A股和港股)
            if market == ['A', 'HK']:
                try:
                    df = self.get_tushare_data(symbol, freq[0].upper(), adj, start_date, end_date)
                    if df is not None and not df.empty:
                        print("[INFO] tushare成功")
                        return df
                    tried.append('tushare')
                except Exception as e:
                    print(f"[WARN] tushare失败: {e}")
                    tried.append('tushare')
        elif market == 'US':
            # 使用 akshare 获取美股数据
            try:
                df = self.get_akshare_data(symbol, market, freq, adj, start_date, end_date)
                if df is not None and not df.empty:
                    # 确保数据在请求的时间范围内
                    df = self._ensure_date_range(df, start_date, end_date)
                    print("[INFO] akshare获取美股数据成功")
                    return df
                tried.append('akshare')
            except Exception as e:
                print(f"[WARN] akshare获取美股数据失败: {e}")
                tried.append('akshare')
        else:
            raise ValueError('不支持的市场类型')
            
        raise RuntimeError(f"所有数据源均失败: {tried}")

    def get_akshare_data(self, symbol, market, freq='daily', adj='qfq', 
                 start_date=None, end_date=None):
        import akshare as ak
        import pandas as pd
        try:
            if market == 'A':
                df = ak.stock_zh_a_hist(symbol=symbol, 
                                  period=freq,
                                  start_date=start_date,
                                  end_date=end_date,
                                  adjust=adj)
                if df is None or df.empty:
                    return None
                # 获取A股交易日历
                trade_days = ak.tool_trade_date_hist_sina()
                trade_days = pd.to_datetime(trade_days['trade_date'])
                mask = (trade_days >= pd.to_datetime(start_date)) & (trade_days <= pd.to_datetime(end_date))
                all_days = trade_days[mask]
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').reindex(all_days)
                # 标记停牌日
                df['is_suspended'] = df['open'].isna().astype(int)
                df = df.reset_index().rename(columns={'index': '日期'})
            elif market == 'HK':
                df = ak.stock_hk_daily(symbol=symbol)
                if df is None or df.empty:
                    return None
                # 兼容不同列名
                if 'date' in df.columns:
                    df = df.rename(columns={'date': '日期'})
                if '日期' not in df.columns:
                    raise ValueError("港股数据缺少日期列")
                # 获取港股交易日历（用A股日历近似）
                trade_days = ak.tool_trade_date_hist_sina()
                trade_days = pd.to_datetime(trade_days['trade_date'])
                mask = (trade_days >= pd.to_datetime(start_date)) & (trade_days <= pd.to_datetime(end_date))
                all_days = trade_days[mask]
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期').reindex(all_days)
                df['is_suspended'] = df['open'].isna().astype(int)
                df = df.reset_index().rename(columns={'index': '日期'})
            elif market == 'US':
                df = ak.stock_us_daily(symbol=symbol)
                if df is None or df.empty:
                    return None
                if 'date' in df.columns:
                    df = df.rename(columns={'date': '日期'})
                if '日期' not in df.columns:
                    raise ValueError("美股数据缺少日期列")
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.set_index('日期')
            else:
                raise ValueError(f"不支持的市场类型: {market}")
            return df
        except Exception as e:
            print(f"[WARN] akshare失败: {str(e)}")
            return None