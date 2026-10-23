"""The JAX engine must reproduce the original PyTorch implementation given identical weights."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tramdag.engine import transforms as T  # noqa: E402
from tramdag.engine.checkpoint import params_from_torch_state_dict  # noqa: E402
from tramdag.engine.node_model import continuous_nll_terms, init_node_params, node_forward, node_nll_terms, ordinal_bounds  # noqa: E402
from tramdag.engine.sampling import invert_h, ordinal_from_latent  # noqa: E402
from tramdag.engine.structure import build_node_structure  # noqa: E402
from tramdag.legacy_torch.utils import continous as TC  # noqa: E402
from tramdag.legacy_torch.utils import ordinal as TO  # noqa: E402
from tramdag.legacy_torch.utils.continous import contram_nll  # noqa: E402
from tramdag.legacy_torch.utils.data import GenericDataset  # noqa: E402
from tramdag.legacy_torch.utils.model_helpers import get_fully_specified_tram_model  # noqa: E402
from tramdag.legacy_torch.utils.ordinal import ontram_nll  # noqa: E402

from conftest import make_nodes_conf, mixed_frame  # noqa: E402

DATA_TYPE = {"x1": "continous", "x2": "continous", "x3": "ordinal_Xn_Yo", "x4": "continous", "x5": "ordinal_Xn_Yo"}
ADJ = [
    ["0", "ls", "ls", "cs11", "ci"],
    ["0", "0", "cs", "cs12", "0"],
    ["0", "0", "0", "ls", "0"],
    ["0", "0", "0", "0", "ls"],
    ["0", "0", "0", "0", "0"],
]


@pytest.fixture(scope="module")
def setup():
    df = mixed_frame(n=3000, seed=1)
    nodes = make_nodes_conf(ADJ, DATA_TYPE, levels={"x3": 4, "x5": 2})
    mm = {c: (float(df[c].quantile(0.05)), float(df[c].quantile(0.95))) for c in df}
    torch.manual_seed(0)
    out = {}
    for node in nodes:
        tm = get_fully_specified_tram_model(node, {"nodes": nodes}, device="cpu", verbose=False)
        # perturb weights so intercepts are not trivially ordered / shifts not tiny
        with torch.no_grad():
            for p in tm.parameters():
                p.add_(0.3 * torch.randn_like(p))
        tm.eval()
        st = build_node_structure(node, nodes)
        params = params_from_torch_state_dict(tm.state_dict(), init_node_params(jax.random.PRNGKey(0), st))
        ds = GenericDataset(df, target_col=node, target_nodes=nodes)
        (xi, xs), y = next(iter(torch.utils.data.DataLoader(ds, batch_size=len(df))))
        with torch.no_grad():
            tout = tm(int_input=xi, shift_input=xs)
        out[node] = dict(tm=tm, st=st, params=params, tout=tout, y=y, enc=st.encode_frame(df))
    return df, nodes, mm, out


@pytest.mark.parametrize("node", list(DATA_TYPE))
def test_nll_matches_torch(setup, node):
    _, _, mm, out = setup
    o = out[node]
    if o["st"].target == "continuous":
        ref = contram_nll(o["tout"], o["y"], min_max=torch.tensor(mm[node])).item()
    else:
        ref = ontram_nll(o["tout"], o["y"]).item()
    got = float(jnp.mean(node_nll_terms(o["st"], o["params"], o["enc"], o["enc"]["y"], mm[node])))
    assert got == pytest.approx(ref, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("node", list(DATA_TYPE))
def test_forward_outputs_match(setup, node):
    _, _, _, out = setup
    o = out[node]
    int_out, shift_sum, terms = node_forward(o["st"], o["params"], o["enc"])
    np.testing.assert_allclose(np.asarray(int_out), o["tout"]["int_out"].numpy(), rtol=1e-5, atol=1e-5)
    ref_shift = sum(t.numpy()[:, 0] for t in o["tout"]["shift_out"]) if o["tout"]["shift_out"] else 0 * np.asarray(shift_sum)
    np.testing.assert_allclose(np.asarray(shift_sum), ref_shift, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("node", ["x2", "x4"])
def test_continuous_latent_matches(setup, node):
    _, _, mm, out = setup
    o = out[node]
    h_ref, _ = contram_nll(o["tout"], o["y"], min_max=torch.tensor(mm[node]), return_h=True)
    int_out, shift_sum, _ = node_forward(o["st"], o["params"], o["enc"])
    _, h = continuous_nll_terms(int_out, shift_sum, o["enc"]["y"], *mm[node])
    np.testing.assert_allclose(np.asarray(h), h_ref.numpy(), rtol=1e-5, atol=1e-4)


def test_ordinal_latent_interval_matches(setup):
    _, _, _, out = setup
    o = out["x3"]
    trans = TO.transform_intercepts_ordinal(o["tout"]["int_out"])
    shift = torch.stack(o["tout"]["shift_out"], dim=1).sum(dim=1).view(-1)
    cut = (trans - shift.unsqueeze(1)).numpy()
    k = o["enc"]["y"]
    int_out, shift_sum, _ = node_forward(o["st"], o["params"], o["enc"])
    lo, up = ordinal_bounds(int_out, shift_sum, jnp.asarray(k))
    np.testing.assert_allclose(np.asarray(lo), cut[np.arange(len(k)), k], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(up), cut[np.arange(len(k)), k + 1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("node", ["x1", "x2", "x4"])
def test_continuous_sampling_inverse_matches_torch_root_finder(setup, node):
    _, _, mm, out = setup
    o = out[node]
    thetas_t = TC.transform_intercepts_continous(o["tout"]["int_out"])
    shifts_t = torch.stack(o["tout"]["shift_out"], 0).sum(0).squeeze(-1) if o["tout"]["shift_out"] else torch.zeros(len(thetas_t))
    rng = np.random.default_rng(0)
    u = torch.tensor(rng.logistic(size=len(thetas_t)), dtype=torch.float32)
    k_min, k_max = torch.tensor(mm[node][0]), torch.tensor(mm[node][1])
    f = lambda x: TC.vectorized_object_function(thetas_t, x, shifts_t, u, k_min, k_max)
    ref = TC.chandrupatla_root_finder(f, torch.full_like(u, -1e5), torch.full_like(u, 1e5), max_iter=100, tol=1e-12).numpy()
    int_out, shift_sum, _ = node_forward(o["st"], o["params"], o["enc"])
    got = np.asarray(invert_h(T.transform_intercepts_continuous(int_out), shift_sum, jnp.asarray(u.numpy()), *mm[node]))
    # the JAX root solves h(x) = u (closed form in the linear tails, bisection inside) ...
    res_jax = np.abs(f(torch.tensor(got)).numpy())
    assert res_jax.max() < 1e-4
    # ... and agrees with the original solver wherever that one converged. (The original Chandrupatla
    # search starts from [-1e5, 1e5] in float32 and stops after 100 iterations; in the extrapolated
    # tails it can stop with residuals of ~1e-2, i.e. the JAX samples are the more accurate ones.)
    ok = np.abs(f(torch.tensor(ref)).numpy()) < 1e-5
    assert ok.mean() > 0.5
    scale = mm[node][1] - mm[node][0]
    np.testing.assert_allclose(got[ok], ref[ok], atol=1e-4 * scale, rtol=1e-4)


def test_ordinal_sampling_matches_torch_rule(setup):
    _, _, _, out = setup
    o = out["x3"]
    trans = TO.transform_intercepts_ordinal(o["tout"]["int_out"])
    shift = torch.stack(o["tout"]["shift_out"], dim=1).sum(dim=1).view(-1, 1)
    hh = trans - shift
    u = torch.tensor(np.random.default_rng(1).logistic(size=hh.shape[0]), dtype=torch.float32).unsqueeze(1)
    ref = ((u > hh[:, :-1]) & (u <= hh[:, 1:])).float().argmax(dim=1).numpy()
    int_out, shift_sum, _ = node_forward(o["st"], o["params"], o["enc"])
    got = np.asarray(ordinal_from_latent(int_out, shift_sum, jnp.asarray(u.numpy()[:, 0])))
    np.testing.assert_array_equal(got, ref)


def test_transform_inverses():
    rng = np.random.default_rng(0)
    tt = rng.normal(size=(5, 20)).astype(np.float32)
    th = np.asarray(T.transform_intercepts_continuous(jnp.asarray(tt)))
    np.testing.assert_allclose(th, TC.transform_intercepts_continous(torch.tensor(tt)).numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(T.inverse_transform_intercepts_continuous(th), tt, atol=1e-4)
    oi = rng.normal(size=(5, 3)).astype(np.float32)
    cut = np.asarray(T.ordinal_cutpoints(jnp.asarray(oi)))
    np.testing.assert_allclose(cut, TO.transform_intercepts_ordinal(torch.tensor(oi)).numpy()[:, 1:-1], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(T.inverse_transform_intercepts_ordinal(cut), oi, atol=1e-4)
