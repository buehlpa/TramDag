# CLAUDE.md

TRAM-DAG: interpretable neural causal models (paper: arXiv 2503.16206). Every node of a DAG gets a
transformation model `h(x_j | pa(x_j))` built from a simple or complex intercept plus linear (`ls`) or
complex (`cs`) shift terms. Continuous targets use a Bernstein polynomial (CONTRAM), ordinal targets
cut-points (ONTRAM). The fitted SCM supports observational, interventional (`do`) and counterfactual sampling.

## Environment

- Use the project venv: `.venv/bin/python`, `.venv/bin/pytest` (Python 3.13, JAX with CUDA 12, torch CPU).
  Recreate with `python -m venv .venv && .venv/bin/pip install -e ".[dev,cuda]"`.
- `import tramdag` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` (setdefault). Under WSL JAX may still print
  harmless `CUDA_ERROR_OUT_OF_MEMORY` allocator lines; filter them with `grep -v "^E1"`.
- R is not installed here, so `set_initial_weights=True` (R warm start via `tramdag/utils/r_subprocess.py`)
  cannot be run; the parameter placement is unit-tested with fake R output.

## Commands

```bash
.venv/bin/pytest -q                                        # ~1 min, 55 tests (needs GPU or CPU JAX, torch)
.venv/bin/python benchmarks/replicate_examples.py          # paper experiments, ~25 min incl. torch timing
.venv/bin/python benchmarks/replicate_examples.py --quick  # 1 seed, no torch, ~1 min
.venv/bin/python benchmarks/replicate_examples.py --from-json   # re-render REPLICATION.md only
.venv/bin/python benchmarks/ui_e2e.py <workdir>            # editor in a real JupyterLab via playwright
.venv/bin/python -m tramdag.ui path/to/configuration.json --data train.csv   # browser editor
JAX_PLATFORMS=cpu .venv/bin/python -m pdoc tramdag tramdag.model tramdag.dag_spec tramdag.engine tramdag.ui tramdag.legacy_torch -o docs
```

## Layout

- `tramdag/model.py` – public `TramDagModel` (JAX). API mirrors the original torch class
  (`from_config`, `from_directory`, `fit`, `sample`, `get_latent`, `nll`, `summary`, `plot_*`) plus
  `coefficients()`, `shift_term()`, `transformation()`, `cutpoints()`, `formulas()`.
- `tramdag/engine/` – JAX core
  - `transforms.py` Bernstein basis, `h`, intercept transforms (formulas copied 1:1 from the torch code)
  - `nets.py` network registry (`register_model`, inline `"MLP(32,32,tanh)"`), torch-default init
  - `structure.py` config node -> `NodeStructure` (term ordering/grouping identical to the original), data encoding
  - `node_model.py` forward pass, CONTRAM / ONTRAM per-sample NLL, ordinal latent intervals
  - `training.py` `JointTrainer`: all nodes in one jit, epochs in `lax.scan`, best-model tracking on device
  - `sampling.py` `DagSampler`: closed-form tail inversion + bisection, truncated-logistic CF draws
  - `checkpoint.py` `.npz` params, torch `state_dict` -> JAX conversion
- `tramdag/TramDagConfig.py` – configuration file wrapper. Original widget workflow (`set_data_type`,
  `set_meta_adj_matrix`, ...) plus new `from_dataframe`, `from_edges`, `from_spec`, `add_edges`,
  `add_edge`, `remove_edge`, `set_node`, `set_training`, `describe`, `edit`.
- `tramdag/dag_spec.py` – `DagSpec` (nodes/edges/groups), edge mini-language, validation, conversion to and
  from the `configuration.json` format, data type inference, auto layout.
- `tramdag/ui/` – visual editor: `static/editor.js` (dependency-free ES module, anywidget API),
  `widget.py` (anywidget `DagEditor`), `server.py` (stdlib HTTP server, same JS), `backend.py` (shared
  validation/save logic – keep all rules in Python, the JS only edits JSON).
- `tramdag/utils/configuration.py`, `r_subprocess.py` – shared helpers (also used by legacy).
- `tramdag/legacy_torch/` – the original PyTorch implementation, kept as the parity reference. Do not
  change its numerics; tests compare against it.
- `tests/` – parity (`test_parity_torch.py`), engine, config, UI. `benchmarks/` – replication + UI e2e.
- `example_notebooks/` – `0_quickstart_jax.ipynb` (new) and the original paper notebooks (run with torch;
  their committed `*/` folders hold the torch training histories used as reference in the replication).

## Configuration format (unchanged, shared by both engines)

`configuration.json`: `data_type` (in causal order), upper-triangular `adj_matrix` of term codes,
`model_names`, and per node `parents`, `parents_datatype`, `transformation_terms_in_h()`,
`transformation_term_nn_models_in_h()`, `levels` (ordinal). Optional new keys: `training`
(fit defaults) and `ui_layout` (editor positions). Term codes: `ls`, `cs`, `ci`, numbered interaction
groups `csGP` / `ciGP` (group G, participant P, both 1–9). Data types: `continous` (sic, keep the
spelling), `ordinal_Xn_Yo`, `ordinal_Xc_Yo`, `ordinal_Xn_Yc`, `ordinal_Xc_Yc`. `Xn` = one-hot as
predictor, `Yo` = ordinal likelihood.

## Conventions and gotchas

- Sign convention differs: continuous `h = h_I(y) + sum(shifts)`, ordinal `P(Y<=k) = sigmoid(theta_k - sum(shifts))`.
- Kept on purpose for compatibility: `h_dag` uses `mean` over Bernstein terms but `h_dag_dash` is an
  un-normalised sum, so continuous NLLs inside the scaling range carry a constant `-log(20/19)`.
  Changing this alters reported NLLs vs. the paper/notebooks.
- Scaling for continuous targets = 5% / 95% training quantiles (`min_max_scaling.json`).
- Trained Bernstein coefficients are O(10–100); Adam (lr 0.01) needs thousands of steps. Small test
  fits must use small batches / many epochs or coefficients will look wrong.
- `fit()` resumes from `last_params.npz` + histories when present (`resume=False` to start over).
  Torch-only kwargs (`num_workers`, `train_mode`, ...) are accepted and ignored.
- Inference (`nll`, `get_latent`, `sample`) uses the best (lowest val NLL) parameters by default (`state=`).
- Avoid recompilation: the sampler is cached on the model and pads inputs to power-of-two buckets;
  host-side constants (e.g. binomial coefficients) are precomputed with NumPy because `gammaln` is slow
  to compile on GPU.
- Original sampler (legacy) leaves root-finding residuals up to ~0.5 in 2–4% of tail samples; the JAX
  sampler is exact to ~1e-5. Parity tests therefore compare only where the torch root converged.
- Commits: work on a branch, never commit to `main` without being asked.
