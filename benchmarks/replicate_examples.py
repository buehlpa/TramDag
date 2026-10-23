"""
Replicate the example notebooks (paper sections 6.1 / 6.2) with the JAX engine.

For every experiment the script

1. regenerates the data exactly like the notebook (seed 42, 80/10/10 split),
2. trains the JAX model with the notebook settings for several seeds,
3. compares the final NLL, linear shifts and intercepts with the numbers printed in the
   notebooks, the stored PyTorch training histories, the R reference fits quoted in the
   notebook comments and the true DGP parameters,
4. checks observational / interventional / counterfactual sampling against the DGP,
5. times the original PyTorch engine on the same data (a few epochs, CPU).

Usage::

    python benchmarks/replicate_examples.py            # full run
    python benchmarks/replicate_examples.py --quick    # 1 seed, no torch timing

Results go to ``benchmarks/results/replication.json`` and ``benchmarks/REPLICATION.md``.
"""

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from dgps import complex_dgp, linear_dgp, mixed_dgp, split  # noqa: E402

NB = os.path.join(ROOT, "example_notebooks")
R_COLR_X2 = 1.973891  # Colr(x2 ~ x1) quoted in the notebook comment (same x1, x2 in all experiments)

EXPERIMENTS = [
    dict(
        name="6.1 linear DGP, linear shifts",
        key="linear_DGP_LS",
        dgp=linear_dgp,
        config=f"{NB}/Continous_3_vars_linear_DGP_LS/configuration.json",
        epochs=200,
        intervention={"x1": 0.0},
        notebook_nll={"x1": -0.5739832321802775, "x2": 0.3467276295026143, "x3": 2.418768564860026},
        truth={"x2": {"ls(x1)": 2.0}, "x3": {"ls(x1)": -0.2, "ls(x2)": 0.3}},
        r_ref={"x2": {"ls(x1)": R_COLR_X2}},
    ),
    dict(
        name="6.1 linear DGP, complex shift cs(x1)",
        key="linear_DGP_CS",
        dgp=linear_dgp,
        config=f"{NB}/Continous_3_vars_linear_DGP_CS/configuration.json",
        epochs=400,
        intervention={"x1": 0.0},
        notebook_nll=None,  # the notebook printed NLLs of a different run (x1 = 2.12), see report
        truth={"x2": {"ls(x1)": 2.0}, "x3": {"ls(x2)": 0.3}},
        r_ref={"x2": {"ls(x1)": R_COLR_X2}},
    ),
    dict(
        name="6.1 complex DGP, complex shift cs(x2)",
        key="complex_DGP_CS",
        dgp=complex_dgp,
        config=f"{NB}/Continous_3_vars_complex_DGP_LS/configuration.json",
        epochs=300,
        intervention={"x1": 0.0},
        notebook_nll={"x1": -0.5735894640286764, "x2": 0.3474327325820923, "x3": 2.419990301132202},
        truth={"x2": {"ls(x1)": 2.0}, "x3": {"ls(x1)": -0.2}},
        r_ref={"x2": {"ls(x1)": R_COLR_X2}},
        complex_truth=("x3", "x2", lambda x: 0.75 * np.arctan(5 * (x + 0.12))),
    ),
    dict(
        name="6.2 mixed DGP (x3 ordinal, 4 levels), linear shifts",
        key="mixed_DGP_LS",
        dgp=lambda n, doX=(None, None, None), seed=None: mixed_dgp(n, doX=doX, seed=seed if seed and seed > 0 else None),
        config=None,
        epochs=200,
        intervention={"x1": -3.0},
        notebook_nll={"x1": -0.5743332703908285, "x2": 0.34419193863868713, "x3": 1.212408224741618},
        truth={"x2": {"ls(x1)": 2.0}, "x3": {"ls(x1)": -0.2, "ls(x2)": -1.0}},
        r_ref={"x2": {"ls(x1)": R_COLR_X2}, "x3": {"ls(x1)": -0.1815344, "ls(x2)": -1.0012274}},
        r_thetas={"x3": [-1.998953, 0.426397, 1.032376]},
        r_nll={"x3": 194022.51 / (2 * 80_000)},
        true_thetas={"x3": [-2.0, 0.42, 1.02]},
    ),
]


def mixed_config():
    from tramdag.utils.configuration import create_nn_model_names, create_node_dict

    dt = {"x1": "continous", "x2": "continous", "x3": "ordinal_Xn_Yo"}
    adj = np.array([["0", "ls", "ls"], ["0", "0", "ls"], ["0", "0", "0"]], dtype=object)
    nn = create_nn_model_names(adj, dt)
    nodes = create_node_dict(adj, nn, dt, None, None, levels_dict={"x3": 4})
    return {
        "experiment_name": "Mixed_3_vars_linear_DGP_LS",
        "PATHS": {"DATA_PATH": None, "LOG_DIR": None, "EXPERIMENT_DIR": None},
        "data_type": dt,
        "adj_matrix": adj.tolist(),
        "model_names": nn.tolist(),
        "seed": None,
        "nodes": nodes,
    }


def make_run_dir(exp, base, tag):
    run = os.path.join(base, f"{exp['key']}_{tag}")
    shutil.rmtree(run, ignore_errors=True)
    os.makedirs(run)
    conf = json.load(open(exp["config"])) if exp["config"] else mixed_config()
    conf["PATHS"]["EXPERIMENT_DIR"] = run
    with open(os.path.join(run, "configuration.json"), "w") as f:
        json.dump(conf, f, indent=2)
    return run


def stored_torch_results(exp):
    """Final values of the PyTorch run whose histories are committed in example_notebooks/."""
    if not exp["config"]:
        return None
    d = os.path.dirname(exp["config"])
    out = {}
    for node in ("x1", "x2", "x3"):
        tr = json.load(open(f"{d}/{node}/train_loss_hist.json"))
        va = json.load(open(f"{d}/{node}/val_loss_hist.json"))
        r = {"epochs": len(tr), "train_last": tr[-1], "val_last": va[-1]}
        p = f"{d}/{node}/linear_shifts_all_epochs.json"
        if os.path.exists(p):
            ls = json.load(open(p))
            r["ls"] = ls[list(ls)[-1]]
        out[node] = r
    return out


def ks(a, b):
    return float(ks_2samp(np.asarray(a, float), np.asarray(b, float)).statistic)


def tv(a, b, levels):
    pa = np.bincount(np.asarray(a, int), minlength=levels) / len(a)
    pb = np.bincount(np.asarray(b, int), minlength=levels) / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def true_counterfactual_mixed(obs, x2_cf, rng, n_draws):
    """Ground-truth CF distribution of x3 under do(x2 = x2_cf) for the 6.2 DGP."""
    theta = np.array([-2.0, 0.42, 1.02])
    h = theta[None, :] + 0.2 * obs["x1"].to_numpy()[:, None] + obs["x2"].to_numpy()[:, None]
    k = obs["x3"].to_numpy().astype(int)
    padded = np.concatenate([np.full((len(obs), 1), -np.inf), h, np.full((len(obs), 1), np.inf)], 1)
    lo, up = padded[np.arange(len(obs)), k], padded[np.arange(len(obs)), k + 1]
    F = lambda x: 1 / (1 + np.exp(-x))
    v = rng.uniform(size=(len(obs), n_draws))
    p = np.clip(F(lo)[:, None] + v * (F(up) - F(lo))[:, None], 1e-12, 1 - 1e-12)
    u = np.log(p) - np.log1p(-p)
    h_cf = theta[None, :] + 0.2 * obs["x1"].to_numpy()[:, None] + x2_cf
    x3_cf = (u[:, :, None] > h_cf[:, None, :]).sum(-1)
    return np.stack([(x3_cf == j).mean(1) for j in range(4)], 1)


def run_jax(exp, data, base, seeds, device="auto", verbose=False):
    from tramdag import TramDagConfig, TramDagModel

    train, val, test = data
    runs = []
    model0 = None
    for seed in seeds:
        run = make_run_dir(exp, base, f"jax_seed{seed}_{device}")
        cfg = TramDagConfig.load_json(os.path.join(run, "configuration.json"))
        t0 = time.perf_counter()
        m = TramDagModel.from_config(cfg, seed=seed, device=device)
        m.fit(train, val, epochs=exp["epochs"], batch_size=1000, learning_rate=1e-2, verbose=verbose, seed=seed)
        fit_s = time.perf_counter() - t0
        r = {
            "seed": seed,
            "fit_seconds": fit_s,
            "nll_train_last": m.nll(train, state="last"),
            "nll_test_best": m.nll(test, state="best"),
            "val_best": {n: m.get_train_val_nll(n, "best")[1] for n in m.order},
            "ls": {n: m.get_linear_shifts(n, "last") for n in m.order},
            "thetas": {n: [t[0] for t in m.get_thetas(n, "last")] for n in m.order if m.structures[n].target == "ordinal"},
        }
        if "complex_truth" in exp:
            node, parent, f = exp["complex_truth"]
            grid = np.linspace(np.quantile(train[parent], 0.05), np.quantile(train[parent], 0.95), 200)
            est = m.shift_term(node, parent, grid)
            # the shift is identified up to a constant (absorbed by the intercept) -> compare centred
            tru = -f(grid) / 1.0  # h = 0.63 x3 - 0.2 x1 - f(x2)  => shift = -f(x2)
            r["complex_shift_rmse_centered"] = float(np.sqrt(np.mean(((est - est.mean()) - (tru - tru.mean())) ** 2)))
        runs.append(r)
        if model0 is None:
            model0 = m
    return runs, model0


def sampling_checks(exp, m, data, rng):
    train, val, test = data
    out = {}
    t0 = time.perf_counter()
    s, _ = m.sample(number_of_samples=10_000, seed=1, save=False)
    out["observational_seconds"] = time.perf_counter() - t0
    obs = {}
    for n in m.order:
        if m.structures[n].target == "ordinal":
            obs[n] = {"tv_vs_test": tv(s[n], test[n], m.structures[n].levels)}
        else:
            obs[n] = {"ks_vs_test": ks(s[n], test[n]), "mean": float(np.mean(s[n])), "test_mean": float(test[n].mean()),
                      "std": float(np.std(s[n])), "test_std": float(test[n].std())}
    out["observational"] = obs

    do = exp["intervention"]
    s_do, _ = m.sample(do_interventions=do, number_of_samples=10_000, seed=2, save=False)
    doX = [do.get("x1"), None, None]
    truth = exp["dgp"](10_000, doX=doX)
    inter = {}
    for n in m.order:
        if n in do:
            continue
        if m.structures[n].target == "ordinal":
            inter[n] = {"tv_vs_dgp": tv(s_do[n], truth[n], m.structures[n].levels)}
        else:
            inter[n] = {"ks_vs_dgp": ks(s_do[n], truth[n]), "mean": float(np.mean(s_do[n])), "dgp_mean": float(truth[n].mean())}
    out["interventional"] = {"do": do, "nodes": inter}

    sub = train.iloc[:100]
    u = m.get_latent(sub)
    s_cf, _ = m.sample(predefined_latent_samples_df=u, seed=3, save=False, number_of_counterfactual_samples=2000)
    cons = {}
    for n in m.order:
        if m.structures[n].target == "ordinal":
            p_obs = s_cf[n][np.arange(len(sub)), sub[n].to_numpy().astype(int)]
            cons[n] = {"mean_prob_of_observed_class": float(np.mean(p_obs))}
        else:
            v = s_cf[n] if not isinstance(s_cf[n], dict) else None
            if v is not None:
                cons[n] = {"max_abs_reconstruction_error": float(np.max(np.abs(v - sub[n].to_numpy())))}
    out["counterfactual_consistency"] = cons

    if exp["key"] == "mixed_DGP_LS":
        s_q, _ = m.sample(do_interventions={"x2": -0.2}, predefined_latent_samples_df=u, seed=4, save=False,
                          number_of_counterfactual_samples=5000)
        true_q = true_counterfactual_mixed(sub, -0.2, rng, 5000)
        out["counterfactual_query_x2=-0.2"] = {
            "mean_tv_model_vs_true_cf_distribution": float(np.mean(0.5 * np.abs(s_q["x3"] - true_q).sum(1))),
            "model_avg_class_probs": [round(float(v), 4) for v in s_q["x3"].mean(0)],
            "true_avg_class_probs": np.round(true_q.mean(0), 4).tolist(),
        }
    return out


def torch_timing(exp, data, base, epochs=2):
    """Seconds per epoch (all nodes) and sampling time of the original engine on CPU."""
    import torch  # noqa: F401
    from tramdag import TramDagConfig
    from tramdag.legacy_torch import TramDagModel as TorchModel

    train, val, _ = data
    run = make_run_dir(exp, base, "torch")
    cfg = TramDagConfig.load_json(os.path.join(run, "configuration.json"))
    m = TorchModel.from_config(cfg, device="cpu")
    t0 = time.perf_counter()
    m.fit(train, val, epochs=epochs, batch_size=1000, learning_rate=1e-2, device="cpu", verbose=False,
          num_workers=8, prefetch_factor=8, persistent_workers=True)
    per_epoch = (time.perf_counter() - t0) / epochs
    t0 = time.perf_counter()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m.sample(number_of_samples=10_000, verbose=False)
    sample_s = time.perf_counter() - t0
    return {"seconds_per_epoch_all_nodes": per_epoch, "sampling_10k_seconds": sample_s, "timed_epochs": epochs}


def summarize(results):
    lines = ["# Replication of the example notebooks with the JAX engine", ""]
    lines.append("Generated by `benchmarks/replicate_examples.py`. Same data (seed 42, 80/10/10 split), "
                 "same configuration, same optimiser settings (Adam, lr 0.01, batch 1000) as the notebooks.")
    lines.append("")
    env = results["environment"]
    lines.append(f"Environment: JAX {env['jax']} on `{env['jax_device']}`, torch {env.get('torch', '-')} on CPU.")
    lines.append("")
    for r in results["experiments"]:
        lines.append(f"## {r['name']}")
        lines.append("")
        seeds = r["jax_runs"]
        nodes = list(seeds[0]["nll_train_last"].keys())
        lines.append("| node | JAX train NLL (mean ± sd over seeds) | notebook (torch) | stored torch history (last train) |")
        lines.append("|---|---|---|---|")
        for n in nodes:
            v = np.array([s["nll_train_last"][n] for s in seeds])
            nb = r.get("notebook_nll") or {}
            st = (r.get("stored_torch") or {}).get(n, {})
            fmt = lambda x: "–" if x is None else f"{x:.4f}"
            lines.append(f"| {n} | {v.mean():.4f} ± {v.std():.4f} | {fmt(nb.get(n))} | {fmt(st.get('train_last'))} |")
        lines.append("")
        lines.append("| coefficient | JAX (mean ± sd) | stored torch | R reference | true DGP |")
        lines.append("|---|---|---|---|---|")
        for n in nodes:
            for k in seeds[0]["ls"][n]:
                v = np.array([s["ls"][n][k] for s in seeds], float)
                st = (r.get("stored_torch") or {}).get(n, {}).get("ls", {}).get(k)
                rr = (r.get("r_ref") or {}).get(n, {}).get(k)
                tr = (r.get("truth") or {}).get(n, {}).get(k)
                f = lambda x: "–" if x is None else f"{x:.4f}"
                lines.append(f"| {n}: {k} | {v.mean():.4f} ± {v.std():.4f} | {f(st)} | {f(rr)} | {f(tr)} |")
        if seeds[0]["thetas"]:
            for n, _ in seeds[0]["thetas"].items():
                v = np.array([s["thetas"][n] for s in seeds])
                lines.append(f"| {n}: cut-points | {np.round(v.mean(0), 4).tolist()} | – | {r['r_thetas'][n]} | {r['true_thetas'][n]} |")
        lines.append("")
        if "complex_shift_rmse_centered" in seeds[0]:
            v = np.array([s["complex_shift_rmse_centered"] for s in seeds])
            lines.append(f"Learned complex shift vs. true `-f(x2)` (centred RMSE over the 5–95% range of x2): {v.mean():.4f} ± {v.std():.4f}")
            lines.append("")
        sc = r["sampling"]
        lines.append("Sampling checks (10 000 samples):")
        lines.append("")
        lines.append("- observational vs. test set: " + ", ".join(
            f"{n} " + (f"KS={d['ks_vs_test']:.3f}" if "ks_vs_test" in d else f"TV={d['tv_vs_test']:.3f}")
            for n, d in sc["observational"].items()))
        lines.append(f"- interventional do{sc['interventional']['do']} vs. DGP: " + ", ".join(
            f"{n} " + (f"KS={d['ks_vs_dgp']:.3f}" if "ks_vs_dgp" in d else f"TV={d['tv_vs_dgp']:.3f}")
            for n, d in sc["interventional"]["nodes"].items()))
        lines.append("- counterfactual consistency (abduct U of 100 training rows, re-sample without intervention): " + ", ".join(
            f"{n} " + (f"max |x - x_cf| = {d['max_abs_reconstruction_error']:.2e}" if "max_abs_reconstruction_error" in d
                       else f"P(observed class) = {d['mean_prob_of_observed_class']:.3f}")
            for n, d in sc["counterfactual_consistency"].items()))
        if "counterfactual_query_x2=-0.2" in sc:
            q = sc["counterfactual_query_x2=-0.2"]
            lines.append(f"- counterfactual query do(x2=-0.2) for 100 rows: mean TV distance to the true CF distribution "
                         f"= {q['mean_tv_model_vs_true_cf_distribution']:.3f} (avg class probs model {q['model_avg_class_probs']} vs true {q['true_avg_class_probs']})")
        lines.append("")
        t = r.get("timing", {})
        jt = np.array([s["fit_seconds"] for s in seeds])
        lines.append("| timing | value |")
        lines.append("|---|---|")
        lines.append(f"| JAX fit, {r['epochs']} epochs, all nodes ({env['jax_device']}) | {jt.mean():.1f} s (incl. compilation) |")
        if "jax_cpu_fit_seconds" in t:
            lines.append(f"| JAX fit, {r['epochs']} epochs, all nodes (CPU) | {t['jax_cpu_fit_seconds']:.1f} s |")
        if "torch" in t:
            tt = t["torch"]
            lines.append(f"| torch fit, measured {tt['timed_epochs']} epochs, extrapolated to {r['epochs']} (CPU) | {tt['seconds_per_epoch_all_nodes'] * r['epochs']:.0f} s |")
            lines.append(f"| speed-up of a full fit | {tt['seconds_per_epoch_all_nodes'] * r['epochs'] / jt.mean():.0f}x |")
            lines.append(f"| sampling 10 000 obs. (JAX first call incl. compilation / torch) | {sc['observational_seconds']:.2f} s / {tt['sampling_10k_seconds']:.1f} s |")
        lines.append("")
    if results.get("notes"):
        lines.append("## Notes")
        lines.append("")
        lines += [f"- {n}" for n in results["notes"]]
        lines.append("")
    return "\n".join(lines)


NOTES = [
        "The original continuous NLL uses h = mean_k(B_k theta_k) but h' = sum_k(B'_k dtheta_k), i.e. the derivative "
        "is 20/19 times too large inside the scaling range [5%, 95% quantile]. Every observation inside that range "
        "therefore gets a constant -log(20/19) = -0.051 in its NLL (the tails are consistent). The term does not "
        "depend on the parameters, so gradients and fitted models are unaffected; the JAX engine keeps it for "
        "comparability with the published numbers.",
        "Notebook 'Continous_3_vars_linear_DGP_CS' printed td_model.nll() values (x1 = 2.12, x2 = 3.07) that do not "
        "match its own stored training histories (x1 = -0.57, x2 = 0.35); they come from an earlier/different model "
        "state, so that experiment is compared against the stored histories instead.",
        "R reference values are quoted from the comments in Mixed_3_vars_linear_DGP_LS.ipynb (Colr / polr fits). "
        "x1 and x2 are generated identically in all four experiments, so Colr(x2 ~ x1) applies to each of them.",
        "Sampling: the original engine inverts h with 100 float32 Chandrupatla iterations started from [-1e5, 1e5]. "
        "On the trained 6.1 models 2-4% of the samples stop with a residual |h(x) - u| > 1e-3 (max 0.5), which moves "
        "those samples by up to 3.5% of the variable range (mean 1e-4). The JAX engine inverts the linear tails in "
        "closed form and bisects the Bernstein part (residual < 1e-5), see tests/test_parity_torch.py.",
        "Coefficient differences between JAX, torch and R are within the sampling uncertainty: the standard error of "
        "ls(x1) in h(x2|x1) is about sqrt(3 / (n Var(x1))) = 0.024 for n = 80 000, and Adam with a constant learning "
        "rate of 0.01 leaves the last iterate fluctuating at that scale. R's Colr also uses a different Bernstein basis "
        "(order 6 on the data range), so it is a reference, not the identical model.",
        "Torch timings use the original code path of the notebooks (DataLoader with per-row __getitem__, "
        "num_workers=8) on CPU; the notebooks were also run with device='cpu'.",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--torch-epochs", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma separated experiment keys")
    ap.add_argument("--from-json", action="store_true", help="only re-render REPLICATION.md from results/replication.json")
    args = ap.parse_args()
    if args.from_json:
        with open(os.path.join(HERE, "results", "replication.json")) as f:
            results = json.load(f)
        results["notes"] = NOTES
        with open(os.path.join(HERE, "REPLICATION.md"), "w") as f:
            f.write(summarize(results))
        return

    import jax

    base = tempfile.mkdtemp(prefix="tramdag_replication_")
    seeds = [0] if args.quick else list(range(args.seeds))
    results = {"environment": {"jax": jax.__version__, "jax_device": str(jax.devices()[0])}, "experiments": []}
    try:
        import torch

        results["environment"]["torch"] = torch.__version__
    except ImportError:
        pass
    rng = np.random.default_rng(0)
    for exp in EXPERIMENTS:
        if args.only and exp["key"] not in args.only.split(","):
            continue
        print(f"\n=== {exp['name']}")
        data = split(exp["dgp"](100_000, seed=42))
        runs, m = run_jax(exp, data, base, seeds)
        print("  JAX nll:", {k: round(v, 4) for k, v in runs[0]["nll_train_last"].items()}, f"{runs[0]['fit_seconds']:.1f}s")
        res = {k: exp.get(k) for k in ("name", "key", "epochs", "notebook_nll", "truth", "r_ref", "r_thetas", "r_nll", "true_thetas")}
        res["jax_runs"] = runs
        res["stored_torch"] = stored_torch_results(exp)
        res["sampling"] = sampling_checks(exp, m, data, rng)
        res["timing"] = {}
        if not args.quick:
            cpu_runs, _ = run_jax(exp, data, base, [0], device="cpu")
            res["timing"]["jax_cpu_fit_seconds"] = cpu_runs[0]["fit_seconds"]
            try:
                res["timing"]["torch"] = torch_timing(exp, data, base, epochs=args.torch_epochs)
                print("  torch s/epoch:", round(res["timing"]["torch"]["seconds_per_epoch_all_nodes"], 2))
            except Exception as e:  # torch not installed or legacy failure
                res["timing"]["torch_error"] = repr(e)
                print("  torch timing failed:", e)
        results["experiments"].append(res)

    results["notes"] = NOTES
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "replication.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    md = summarize(results)
    with open(os.path.join(HERE, "REPLICATION.md"), "w") as f:
        f.write(md)
    print(md)
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
