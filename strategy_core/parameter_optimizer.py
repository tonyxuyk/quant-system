import numpy as np
from typing import Callable, Dict, Any
from sklearn.model_selection import ParameterGrid
try:
    from skopt import gp_minimize
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

class ParameterOptimizer:
    """参数优化器，支持网格搜索、贝叶斯优化、遗传算法"""
    def __init__(self):
        pass

    def grid_search(self, param_grid: Dict, eval_func: Callable[[Dict], float], top_n=1):
        best = []
        for params in ParameterGrid(param_grid):
            score = eval_func(params)
            best.append((score, params))
        best.sort(reverse=True)
        return best[:top_n]

    def bayesian_opt(self, param_space: list, eval_func: Callable[[list], float], n_calls=20):
        if not SKOPT_AVAILABLE:
            raise ImportError('scikit-optimize未安装')
        res = gp_minimize(eval_func, param_space, n_calls=n_calls)
        return res

    def genetic_opt(self, param_space: Dict, eval_func: Callable[[Dict], float], pop_size=20, n_gen=10, mutation=0.1):
        # 简单遗传算法实现
        keys = list(param_space.keys())
        pop = [
            {k: np.random.choice(param_space[k]) for k in keys}
            for _ in range(pop_size)
        ]
        for gen in range(n_gen):
            scores = [eval_func(ind) for ind in pop]
            idx = np.argsort(scores)[-pop_size//2:]
            survivors = [pop[i] for i in idx]
            children = []
            while len(children) < pop_size - len(survivors):
                p1, p2 = np.random.choice(survivors, 2, replace=False)
                child = {k: np.random.choice([p1[k], p2[k]]) for k in keys}
                # mutation
                if np.random.rand() < mutation:
                    mk = np.random.choice(keys)
                    child[mk] = np.random.choice(param_space[mk])
                children.append(child)
            pop = survivors + children
        best = max(pop, key=eval_func)
        return best 