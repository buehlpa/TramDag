"""Data generating processes of the example notebooks (paper sections 6.1 and 6.2)."""
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import train_test_split


def _x1(n, do):
    if do is None:
        a = np.random.normal(0.25, 0.1, n)
        b = np.random.normal(0.73, 0.05, n)
        return np.where(np.random.choice([0, 1], size=n) == 0, a, b)
    return np.full(n, do, dtype=float)


def continuous_dgp(f):
    """6.1: x1 ~ GMM, x2 = (z2 - 2 x1)/5, x3 = (z3 + 0.2 x1 + f(x2))/0.63."""

    def dgp(n_obs, doX=(None, None, None), seed=-1):
        if seed > 0:
            np.random.seed(seed)
        x1 = _x1(n_obs, doX[0])
        x2 = (logit(np.random.uniform(size=n_obs)) - 2 * x1) / 5 if doX[1] is None else np.full(n_obs, doX[1], float)
        x3 = (logit(np.random.uniform(size=n_obs)) + 0.2 * x1 + f(x2)) / 0.63 if doX[2] is None else np.full(n_obs, doX[2], float)
        return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})

    return dgp


linear_dgp = continuous_dgp(lambda x: -0.3 * x)
complex_dgp = continuous_dgp(lambda x: 0.75 * np.arctan(5 * (x + 0.12)))


def mixed_dgp(n_obs, doX=(None, None, None), seed=None, f=lambda x: x):
    """6.2: x3 ordinal with 4 levels, thresholds (-2, 0.42, 1.02)."""
    if seed is not None:
        np.random.seed(seed)
    x1 = _x1(n_obs, doX[0])
    if doX[1] is None:
        u2 = np.random.uniform(size=n_obs)
        x2 = (np.log(u2 / (1 - u2)) - 2 * x1) / 5.0
    else:
        x2 = np.full(n_obs, doX[1], float)
    if doX[2] is None:
        u3 = np.random.logistic(size=n_obs)
        h = np.array([-2.0, 0.42, 1.02]) + 0.2 * x1.reshape(-1, 1) + f(x2).reshape(-1, 1)
        x3 = np.zeros(n_obs, dtype=int)
        x3[u3 > h[:, 0]] = 1
        x3[u3 > h[:, 1]] = 2
        x3[u3 > h[:, 2]] = 3
    else:
        x3 = np.full(n_obs, doX[2], dtype=int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3.astype(int)})


def split(df):
    train, tmp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(tmp, test_size=0.5, random_state=42)
    return train, val, test
