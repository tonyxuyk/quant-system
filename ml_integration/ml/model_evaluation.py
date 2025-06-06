import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class ModelEvaluation:
    """模型评估类"""
    
    def __init__(self):
        pass
    
    def test_no_future_leakage(self, X, y):
        """测试是否存在未来数据泄露"""
        # 验证所有特征时间戳早于标签时间戳
        for i in range(len(X)):
            if X.index[i] >= y.index[i]:
                raise ValueError("检测到未来数据泄露！")
        return True
    
    def cross_val_sharpe_std(self, trainer, X, y, returns, n=10):
        """
        10次时序CV，夏普比率标准差 < 0.2
        """
        sharpes = []
        tscv = TimeSeriesSplit(n_splits=n)
        for train_idx, test_idx in tscv.split(X):
            trainer.train(X.iloc[train_idx], y.iloc[train_idx])
            pred = trainer.predict(trainer.model, X.iloc[test_idx])
            strat_ret = returns.iloc[test_idx] * pred
            sharpe = strat_ret.mean() / (strat_ret.std() + 1e-9) * (252 ** 0.5)
            sharpes.append(sharpe)
        std = pd.Series(sharpes).std()
        if std >= 0.2:
            print(f"警告: 夏普比率标准差过大: {std}")
        return sharpes
    
    def evaluate_model(self, model, X_test, y_test):
        """基础模型评估"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }