import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def make_nodes_conf(adj, data_type, levels=None, models=None):
    """Build the 'nodes' section of a configuration from an adjacency matrix of term codes."""
    from tramdag.utils.configuration import create_nn_model_names, create_node_dict

    adj = np.array(adj, dtype=object)
    nn = create_nn_model_names(adj, data_type) if models is None else np.array(models, dtype=object)
    return create_node_dict(adj, nn, data_type, None, None, levels_dict=levels)


def mixed_frame(n=4000, seed=0):
    """Toy data with continuous, binary and 4-level ordinal variables."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.5, 0.2, n)
    x2 = (rng.logistic(size=n) - 2 * x1) / 5
    x3 = np.digitize(rng.logistic(size=n) - 0.2 * x1 + x2, [-2.0, 0.4, 1.0])
    x4 = (rng.logistic(size=n) + x3 * 0.5 - x2) / 2
    x5 = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x4 - x1)))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5})


@pytest.fixture
def mixed_df():
    return mixed_frame()
