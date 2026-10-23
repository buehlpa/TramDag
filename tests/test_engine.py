"""Behaviour of the JAX TramDagModel: training, persistence, latents and sampling."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tramdag import TramDagConfig, TramDagModel


def linear_scm(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.5, 0.25, n)
    x2 = (rng.logistic(size=n) - 2.0 * x1) / 5.0          # h(x2|x1) = 5 x2 + 2 x1   -> ls(x1) = 2
    h3 = rng.logistic(size=n)
    x3 = np.digitize(h3 - 0.5 * x1 + 1.5 * x2, [-1.0, 0.5, 1.5])  # ontram: P(x3<=k) = sigmoid(t_k - eta), eta = -0.5 x1 + 1.5 x2
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    exp = tmp_path_factory.mktemp("exp")
    # The Bernstein coefficients of a trained model are O(10-100) and Adam moves them by ~lr per
    # step, so a fit needs several thousand steps (here 60 batches x 150 epochs).
    df = linear_scm(n=14000)
    train, val = df.iloc[:12000], df.iloc[12000:]
    cfg = TramDagConfig.from_dataframe(train, "x1 -> x2; x1, x2 -> x3", experiment_dir=str(exp))
    m = TramDagModel.from_config(cfg, seed=0)
    m.fit(train, val, epochs=150, batch_size=200, learning_rate=0.02, verbose=False)
    return m, train, val, str(exp)


def test_inferred_types(fitted):
    m, *_ = fitted
    assert m.nodes_dict["x3"]["data_type"] == "ordinal_Xn_Yo"
    assert m.nodes_dict["x3"]["levels"] == 4
    assert m.nodes_dict["x1"]["data_type"] == "continous"


def test_recovers_linear_shifts(fitted):
    m, *_ = fitted
    c = m.coefficients().set_index(["node", "input"])["coefficient"]
    assert c[("x2", "x1")] == pytest.approx(2.0, abs=0.25)
    assert c[("x3", "x1")] == pytest.approx(-0.5, abs=0.3)
    assert c[("x3", "x2")] == pytest.approx(1.5, abs=0.4)


def test_files_and_histories(fitted):
    m, _, _, exp = fitted
    for n in ("x1", "x2", "x3"):
        d = os.path.join(exp, n)
        for f in ("initial_params.npz", "best_params.npz", "last_params.npz", "train_loss_hist.json", "val_loss_hist.json"):
            assert os.path.exists(os.path.join(d, f)), f
        tr = json.load(open(os.path.join(d, "train_loss_hist.json")))
        assert len(tr) == 150 and tr[-1] < tr[0]
    ls = json.load(open(os.path.join(exp, "x3", "linear_shifts_all_epochs.json")))
    assert set(ls["epoch_150"]) == {"ls(x1)", "ls(x2)"}
    si = json.load(open(os.path.join(exp, "x3", "simple_intercepts_all_epochs.json")))
    cut = [v[0] for v in si["epoch_150"]]
    assert len(cut) == 3 and cut == sorted(cut)
    assert os.path.exists(os.path.join(exp, "min_max_scaling.json"))


def test_best_is_best(fitted):
    m, _, val, _ = fitted
    hist = m.loss_history()
    for n in m.order:
        best = min(hist[n]["validation"])
        assert m.nll(val, variables=[n], state="best")[n] == pytest.approx(best, abs=2e-4)


def test_from_directory_roundtrip(fitted):
    m, train, _, exp = fitted
    m2 = TramDagModel.from_directory(exp)
    a, b = m.nll(train), m2.nll(train)
    for n in a:
        assert a[n] == pytest.approx(b[n], abs=1e-6)


def test_resume_continues(fitted, tmp_path):
    _, train, val, _ = fitted
    cfg = TramDagConfig.from_dataframe(train, "x1 -> x2", experiment_dir=str(tmp_path / "r"))
    m = TramDagModel.from_config(cfg)
    m.fit(train, val, epochs=5, verbose=False)
    m.fit(train, val, epochs=8, verbose=False)
    hist = m.loss_history()
    assert len(hist["x2"]["train"]) == 8
    m.fit(train, val, epochs=8, verbose=False)  # nothing left to do
    assert len(m.loss_history()["x2"]["train"]) == 8


def test_get_latent_and_reconstruction(fitted):
    m, train, _, _ = fitted
    sub = train.iloc[:50]
    u = m.get_latent(sub)
    assert {"x1_U", "x2_U", "x3_U_lower", "x3_U_upper"} <= set(u.columns)
    assert np.all(u["x3_U_lower"] < u["x3_U_upper"])
    s, lat = m.sample(predefined_latent_samples_df=u, number_of_counterfactual_samples=500, save=False, seed=0)
    np.testing.assert_allclose(s["x1"], sub["x1"], atol=1e-4)
    np.testing.assert_allclose(s["x2"], sub["x2"], atol=1e-4)
    # ordinal node reconstructs the observed class with probability one
    p_obs = s["x3"][np.arange(len(sub)), sub["x3"].to_numpy()]
    np.testing.assert_allclose(p_obs, 1.0)


def test_observational_and_interventional_sampling(fitted):
    m, train, _, exp = fitted
    s, lat = m.sample(number_of_samples=4000, seed=0)
    assert s["x1"].shape == (4000,) and lat["x1"].shape == (4000,)
    assert abs(np.mean(s["x2"]) - train["x2"].mean()) < 0.05
    freq_true = np.bincount(train["x3"], minlength=4) / len(train)
    freq_s = np.bincount(s["x3"].astype(int), minlength=4) / 4000
    assert np.abs(freq_true - freq_s).max() < 0.05
    assert os.path.exists(os.path.join(exp, "x3", "sampling", "sampled.npy"))
    s2, _ = m.sample(do_interventions={"x1": 1.0}, number_of_samples=4000, seed=1, save=False)
    assert np.all(s2["x1"] == 1.0)
    # do(x1=1) lowers x2 (ls = +2 on h, i.e. x2 decreases)
    assert np.mean(s2["x2"]) < np.mean(s["x2"]) - 0.1


def test_counterfactual_distribution_propagates(tmp_path):
    """Ordinal parent with an abducted latent interval -> children become distributions."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.integers(0, 3, n)
    b = (rng.logistic(size=n) - 0.8 * a) / 2
    df = pd.DataFrame({"a": a, "b": b})
    cfg = TramDagConfig.from_dataframe(df, "a -> b", experiment_dir=str(tmp_path / "cf"))
    m = TramDagModel.from_config(cfg)
    m.fit(df, epochs=30, verbose=False)
    u = m.get_latent(df.iloc[:7])
    s, _ = m.sample(predefined_latent_samples_df=u, number_of_counterfactual_samples=300, save=False, seed=0)
    assert s["a"].shape == (7, 3)
    assert isinstance(s["b"], dict) and s["b"]["vals"].shape[0] == 7
    assert np.all(s["b"]["counts"].sum(1) == 300)
    assert m.last_counterfactual_draws["b"].shape == (7, 300)


def test_categorical_and_bool_columns(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    g = pd.Categorical(rng.choice(["lo", "mid", "hi"], n), categories=["lo", "mid", "hi"], ordered=True)
    flag = rng.random(n) < 0.4
    y = rng.normal(size=n) + np.asarray(g.codes) * 0.5 + flag
    df = pd.DataFrame({"g": g, "flag": flag, "y": y})
    cfg = TramDagConfig.from_dataframe(df, "g, flag -> y", experiment_dir=str(tmp_path / "c"))
    assert cfg.conf_dict["nodes"]["g"]["levels"] == 3
    m = TramDagModel.from_config(cfg)
    m.fit(df, epochs=3, verbose=False)
    assert np.isfinite(list(m.nll(df).values())).all()


def test_complex_terms_and_shift_inspection(tmp_path):
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    y = rng.logistic(size=n) / 3 + np.sin(2 * x) + x * z
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    cfg = TramDagConfig.from_dataframe(df, "(x, z) -> y [cs]", experiment_dir=str(tmp_path / "cs"))
    assert cfg.conf_dict["nodes"]["y"]["transformation_terms_in_h()"] == {"x": "cs11", "z": "cs12"}
    m = TramDagModel.from_config(cfg)
    m.fit(df, epochs=4, verbose=False)
    out = m.shift_term("y", "x", np.c_[np.linspace(-1, 1, 5), np.zeros(5)])
    assert out.shape == (5,)
    grid = np.linspace(-3, 3, 7)
    H = m.transformation("y", df.iloc[:2], x=grid)
    assert H.shape == (2, 7) and np.all(np.diff(H, axis=1) > 0)


def test_complex_intercept_and_onehot_parent(tmp_path):
    rng = np.random.default_rng(1)
    n = 2000
    o = rng.integers(0, 4, n)
    x = rng.normal(size=n)
    y = rng.normal(size=n) * (1 + 0.3 * o) + x
    df = pd.DataFrame({"o": o, "x": x, "y": y})
    cfg = TramDagConfig.from_dataframe(df, "o -> y [ci]; x -> y", experiment_dir=str(tmp_path / "ci"))
    m = TramDagModel.from_config(cfg)
    m.fit(df, epochs=3, verbose=False)
    assert m.structures["y"].intercept.n_features == 4  # one-hot parent
    assert "ci(o)" in m.formulas()["y"]


def test_legacy_kwargs_are_accepted(fitted, tmp_path):
    _, train, val, _ = fitted
    cfg = TramDagConfig.from_dataframe(train, "x1 -> x2", experiment_dir=str(tmp_path / "k"))
    m = TramDagModel.from_config(cfg)
    m.fit(train, val, epochs=1, num_workers=8, prefetch_factor=8, persistent_workers=True, verbose=False)
    with pytest.raises(ValueError):
        m.fit(train, epochs=1, not_an_option=1)


def test_warm_start_placement():
    from tramdag.engine import transforms as T
    from tramdag.engine.node_model import init_node_params
    from tramdag.engine.structure import build_node_structure
    from tramdag.model import apply_warm_start
    import jax

    nodes = {
        "a": {"data_type": "continous", "parents": [], "parents_datatype": {}, "transformation_terms_in_h()": {}, "transformation_term_nn_models_in_h()": {}},
        "b": {"data_type": "ordinal_Xn_Yo", "levels": 4, "parents": ["a"], "parents_datatype": {"a": "continous"},
              "transformation_terms_in_h()": {"a": "ls"}, "transformation_term_nn_models_in_h()": {"a": "LinearShift"}},
    }
    st = build_node_structure("b", nodes)
    p = apply_warm_start(init_node_params(jax.random.PRNGKey(0), st), st, [-1.0, 0.3, 2.0], [0.7])
    np.testing.assert_allclose(np.asarray(T.ordinal_cutpoints(p["int"][0]["w"][0])), [-1.0, 0.3, 2.0], atol=1e-5)
    assert float(p["shifts"][0][0]["w"][0, 0]) == pytest.approx(0.7)
    st_a = build_node_structure("a", nodes)
    thetas = np.linspace(-3, 4, 20)
    pa = apply_warm_start(init_node_params(jax.random.PRNGKey(0), st_a), st_a, thetas)
    np.testing.assert_allclose(np.asarray(T.transform_intercepts_continuous(pa["int"][0]["w"][0])), thetas, atol=1e-4)
