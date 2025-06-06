import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Union
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, make_scorer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 可选导入mlflow
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
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
try:
    from skopt import BayesSearchCV
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# 量化指标

def sharpe_ratio(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], returns_series: pd.Series) -> float:
    assert all(y_true.index == returns_series.index), "returns_series index must match y_true index to prevent lookahead bias"
    signals = pd.Series(y_pred, index=y_true.index).shift(1).fillna(0)
    strategy_returns = returns_series * signals
    if strategy_returns.std() == 0:
        return 0.0
    return (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)

def max_drawdown_func(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], returns_series: pd.Series) -> float:
    assert all(y_true.index == returns_series.index), "returns_series index must match y_true index to prevent lookahead bias"
    signals = pd.Series(y_pred, index=y_true.index).shift(1).fillna(0)
    strategy_returns = returns_series * signals
    equity = (1 + strategy_returns).cumprod()
    drawdown = 1 - equity / equity.cummax()
    return drawdown.max()

class MLOptimizer:
    def __init__(self, model_type: str = 'lgb', mlflow_experiment: str = 'ml_opt'):
        assert model_type in ['lgb', 'xgb'], "Initial version only supports lgb/xgb"
        self._check_model_dependencies(model_type)
        self.model_type = model_type
        self.mlflow_experiment = mlflow_experiment
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(mlflow_experiment)
        self.model = None
        self.best_params = None
        self.optimization_history: List[Dict[str, Any]] = []
        self.quant_metrics = {
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown_func
        }

    def _check_model_dependencies(self, model_type: str):
        if model_type == 'lgb' and not LGB_AVAILABLE:
            raise ImportError("lightgbm required. Install with: pip install lightgbm")
        elif model_type == 'xgb' and not XGB_AVAILABLE:
            raise ImportError("xgboost required. Install with: pip install xgboost")
        elif model_type == 'catboost' and not CB_AVAILABLE:
            raise ImportError("catboost required. Install with: pip install catboost")

    def _get_model(self, params: Optional[dict] = None):
        params = params or {}
        model_map = {
            'lgb': lgb.LGBMClassifier(**params) if LGB_AVAILABLE else None,
            'xgb': xgb.XGBClassifier(**params) if XGB_AVAILABLE else None,
            'catboost': cb.CatBoostClassifier(**params) if CB_AVAILABLE else None
        }
        assert self.model_type in model_map, f"Unsupported model_type: {self.model_type}"
        assert model_map[self.model_type] is not None, f"{self.model_type} package not installed"
        return model_map[self.model_type]

    def time_series_cv(self, X: pd.DataFrame, y: pd.Series, params: dict, n_splits: int = 5, scoring: Optional[Callable] = None, n_jobs: int = 1, returns_series: Optional[pd.Series] = None) -> float:
        assert not X.isna().any().any(), "NaN values detected in features"
        assert len(y.unique()) > 1, "Single-class label detected"
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = self._get_model(params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if scoring:
                if returns_series is not None:
                    score = scoring(y_test, y_pred, returns_series.iloc[test_idx])
                else:
                    score = scoring(y_test, y_pred)
            else:
                score = accuracy_score(y_test, y_pred)
            scores.append(score)
        return np.mean(scores)

    def feature_importance(self, X: pd.DataFrame, y: pd.Series, params: Optional[dict] = None) -> pd.Series:
        model = self._get_model(params)
        model.fit(X, y)
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        feat_imp.head(20).plot(kind='bar')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
        return feat_imp

    def optimize(self, X: pd.DataFrame, y: pd.Series, param_grid: dict, n_splits: int = 5, scoring: Optional[Callable] = None, method: str = 'grid', maximize: bool = True, custom_eval_func: Optional[Callable] = None, n_jobs: int = 1, returns_series: Optional[pd.Series] = None) -> tuple:
        best_score = -np.inf if maximize else np.inf
        best_params = None
        history = []
        if method == 'grid':
            grid = ParameterGrid(param_grid)
            for params in grid:
                score = self.time_series_cv(X, y, params, n_splits, scoring=custom_eval_func or scoring, n_jobs=n_jobs, returns_series=returns_series)
                history.append({'params': params, 'score': score})
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params
        elif method == 'bayes' and SKOPT_AVAILABLE:
            if custom_eval_func:
                scorer = make_scorer(lambda yt, yp: custom_eval_func(yt, yp, returns_series.loc[yt.index] if returns_series is not None else None), greater_is_better=maximize)
            else:
                scorer = scoring
            search = BayesSearchCV(self._get_model(), param_grid, n_iter=20, cv=TimeSeriesSplit(n_splits=n_splits), scoring=scorer, n_jobs=n_jobs)
            search.fit(X, y)
            best_score = search.best_score_
            best_params = search.best_params_
            for i, res in enumerate(search.cv_results_['params']):
                history.append({'params': res, 'score': search.cv_results_['mean_test_score'][i]})
        else:
            raise ValueError('method必须为grid或bayes，且需安装scikit-optimize')
        self.best_params = best_params
        self.optimization_history = history
        # mlflow记录
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.log_params(best_params)
                mlflow.log_metric('best_score', best_score)
        return best_params, best_score, history

    def plot_optimization_history(self, metric_name: str = 'score'):
        if not self.optimization_history:
            print('无优化历史')
            return
        scores = [h[metric_name] for h in self.optimization_history]
        plt.figure(figsize=(8, 4))
        plt.plot(scores, marker='o')
        plt.title('Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel(metric_name)
        plt.tight_layout()
        plt.show()

    def fit(self, X: pd.DataFrame, y: pd.Series, params: Optional[dict] = None):
        assert not X.isna().any().any(), "NaN values detected in features"
        assert len(y.unique()) > 1, "Single-class label detected"
        self.model = self._get_model(params)
        self.model.fit(X, y)
        # mlflow保存模型
        if MLFLOW_AVAILABLE:
            with mlflow.start_run():
                mlflow.sklearn.log_model(self.model, 'model')

    def predict(self, X: pd.DataFrame, threshold: float = 0.5, multi_signal: bool = False) -> pd.Series:
        proba = self.model.predict_proba(X)[:, 1]
        if multi_signal:
            # 多空信号：1=买入，-1=卖出，0=观望
            # 这里假设阈值对称，>thresh为1，<1-thresh为-1，其余为0
            long = proba > threshold
            short = proba < (1 - threshold)
            signal = np.zeros_like(proba)
            signal[long] = 1
            signal[short] = -1
            return pd.Series(signal, index=X.index)
        else:
            return pd.Series(np.where(proba > threshold, 1, 0), index=X.index)

    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, metric: Callable = accuracy_score) -> float:
        proba = self.model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 50)
        scores = [metric(y_val, (proba > t).astype(int)) for t in thresholds]
        return thresholds[np.argmax(scores)]

    def get_mlflow_model(self, run_id: str):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available")
        return mlflow.sklearn.load_model(f'runs:/{run_id}/model') 