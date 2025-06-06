import pandas as pd
from ml_models.timeseries_feature_engineering import TimeSeriesFeatureEngineer

def test_no_future_leakage():
    df = load_sample_data()
    engineer = TimeSeriesFeatureEngineer()
    X, y = engineer.preprocess(df)
    # 验证所有特征时间戳早于标签时间戳
    for i in range(len(X)):
        assert X.index[i] < y.index[i], "未来数据泄露！"

def cross_val_sharpe_std(trainer, X, y, returns, n=10):
    """
    10次时序CV，夏普比率标准差 < 0.2
    """
    from sklearn.model_selection import TimeSeriesSplit
    sharpes = []
    tscv = TimeSeriesSplit(n_splits=n)
    for train_idx, test_idx in tscv.split(X):
        trainer.train(X.iloc[train_idx], y.iloc[train_idx])
        pred = trainer.predict(trainer.model, X.iloc[test_idx])
        strat_ret = returns.iloc[test_idx] * pred
        sharpe = strat_ret.mean() / (strat_ret.std() + 1e-9) * (252 ** 0.5)
        sharpes.append(sharpe)
    std = pd.Series(sharpes).std()
    assert std < 0.2, f"夏普比率标准差过大: {std}"
    return sharpes

def load_sample_data():
    # 这里应加载你的样例数据
    # df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
    # return df
    raise NotImplementedError("请实现数据加载函数")