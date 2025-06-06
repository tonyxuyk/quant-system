import pandas as pd
from typing import List, Dict, Any, Callable, Optional
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class StockSelector:
    """选股器，支持规则和机器学习选股"""
    def __init__(self, rules: Optional[List[str]] = None, ml_model: Optional[BaseEstimator] = None):
        self.rules = rules or []
        self.ml_model = ml_model
        self.feature_cols = None
        self.threshold = 0.5

    def set_rules(self, rules: List[str]):
        self.rules = rules

    def rule_select(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = pd.Series([True] * len(df), index=df.index)
        for rule in self.rules:
            # 例：'pe<20', 'roe>15', 'Close>MA20'
            if 'MA' in rule:
                # 处理均线类规则
                import re
                m = re.match(r'(\w+)[<>]=?(\d+)', rule)
                if m:
                    col, val = m.group(1), float(m.group(2))
                    if col.startswith('MA'):
                        window = int(col[2:])
                        ma = df['Close'].rolling(window).mean()
                        if '>' in rule:
                            mask &= df['Close'] > ma
                        else:
                            mask &= df['Close'] < ma
            else:
                # 处理普通规则
                if '<=' in rule:
                    col, val = rule.split('<=')
                    mask &= df[col.strip()] <= float(val)
                elif '>=' in rule:
                    col, val = rule.split('>=')
                    mask &= df[col.strip()] >= float(val)
                elif '<' in rule:
                    col, val = rule.split('<')
                    mask &= df[col.strip()] < float(val)
                elif '>' in rule:
                    col, val = rule.split('>')
                    mask &= df[col.strip()] > float(val)
                elif '==' in rule:
                    col, val = rule.split('==')
                    mask &= df[col.strip()] == float(val)
        return df[mask]

    def fit(self, X: pd.DataFrame, y: pd.Series, features: Optional[List[str]] = None, model: Optional[BaseEstimator] = None):
        """训练机器学习模型，特征支持技术+财务指标"""
        if features is not None:
            X = X[features]
            self.feature_cols = features
        else:
            self.feature_cols = X.columns.tolist()
        if model is not None:
            self.ml_model = model
        elif self.ml_model is None:
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.ml_model.fit(X, y)

    def ml_select(self, X: pd.DataFrame, threshold: float = 0.5, top_n: Optional[int] = None) -> pd.DataFrame:
        """预测上涨概率，返回概率大于阈值的股票"""
        if self.ml_model is None:
            raise ValueError('未设置机器学习模型')
        X_ = X[self.feature_cols] if self.feature_cols else X
        proba = self.ml_model.predict_proba(X_)[:, 1] if hasattr(self.ml_model, 'predict_proba') else self.ml_model.predict(X_)
        result = X.copy()
        result['up_prob'] = proba
        filtered = result[result['up_prob'] >= threshold]
        if top_n:
            filtered = filtered.sort_values('up_prob', ascending=False).head(top_n)
        return filtered

    def select(self, df: pd.DataFrame, mode: str = 'rule', **kwargs) -> pd.DataFrame:
        """统一接口，mode='rule'或'ml'"""
        if mode == 'rule':
            return self.rule_select(df)
        elif mode == 'ml':
            threshold = kwargs.get('threshold', self.threshold)
            top_n = kwargs.get('top_n', None)
            return self.ml_select(df, threshold=threshold, top_n=top_n)
        else:
            raise ValueError('mode必须为rule或ml')

# 示例规则
# selector = StockSelector(rules={
#     'pe_rule': lambda df: df['pe'] < 20,
#     'vol_ratio': lambda df: df['volume_ratio'] > 1.5
# })
# result = selector.rule_select(stock_df) 