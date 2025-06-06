import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# 可选导入
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("警告: mlflow未安装，将跳过实验跟踪功能")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("警告: lightgbm未安装")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("警告: shap未安装，将跳过特征重要性分析")

class ModelTrainer:
    def __init__(self, model_type='lgbm', mlflow_experiment='ml_opt'):
        self.model_type = model_type
        self.model = None
        self.mlflow_experiment = mlflow_experiment
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(mlflow_experiment)

    def train(self, X_train, y_train, early_stopping_rounds=20, n_splits=5, params=None):
        """
        LightGBM时序交叉验证+早停，mlflow记录
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_score = -np.inf
        best_model = None
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm is required for training")
            
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            model = lgb.LGBMClassifier(**(params or {}), n_estimators=1000)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
            score = model.score(X_val, y_val)
            if score > best_score:
                best_score = score
                best_model = model
        self.model = best_model
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.sklearn.log_model(self.model, "model")
        return self.model

    def predict(self, model, X, top_pct=0.1):
        """
        概率转信号，top 10% 买入
        """
        proba = model.predict_proba(X)[:, 1]
        threshold = np.quantile(proba, 1 - top_pct)
        signal = (proba >= threshold).astype(int)
        return pd.Series(signal, index=X.index)

    def feature_importance(self, X):
        """
        SHAP特征重要性
        """
        if not SHAP_AVAILABLE:
            print("SHAP未安装，无法显示特征重要性")
            return
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X)

    def evaluate(self, y_true, y_pred, returns=None):
        """
        信息系数(IC)、分组收益、特征稳定性
        """
        ic = pd.Series(y_pred, index=y_true.index).corr(y_true, method='spearman')
        result = {'IC': ic}
        if returns is not None:
            group_ret = returns.groupby(y_pred).mean()
            result['group_return'] = group_ret
        return result