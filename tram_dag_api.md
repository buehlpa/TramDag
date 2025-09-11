## TramDAG Minimal API (draft)

Input: NumPy/Pandas data frames `df_train`, `df_val`.

### 1) Load or create config (typed; no I/O unless you call .save)
```python
cfg = TramConfig.load("config.yaml") 
```

### 2) Build dataset explicitly 
```python
train = TramDataset.from_dataframe(df_train, cfg)
val   = TramDataset.from_dataframe(df_val,  cfg)
```

### 3) Build the TRAM‑DAG model from configuration
```python
model = TramDAG.from_config(cfg)  # holds all node specs
```

### 4) Train with an explicit trainer; returns history and best model
```python
trainer = Trainer(epochs=100, lr=1e-2, scheduler="cosine", device="auto")
best_model, history = trainer.fit(model, train, val, callbacks=[])
```
- optional: Logging the training process (e.g. with W&B), EarlyStopping, Checkpoint
- Note: warmstarting with given shift / intercept terms should be possible

### 5) Inspect / interpret
```python
print(best_model.summary())                # parameters, structure
print(best_model.get_shift_weights())      # linear shift coefficients if applicable
```

### 6) Persist explicitly (OO style)
```python
best_model.save("runs/best_model.pt")   # model owns its save
history.save("runs/history.json")       # History is a serializable object
cfg.save("runs/config.yaml")
```

### Loading / retraining (OO style)
```python
cfg = TramConfig.load("runs/config.yaml")
model = TramDAG.load("runs/best_model.pt", cfg)  # rebuilds from cfg, loads state
history = TramDag.load_hist("runs/history.json")
```
- Note: retraining from an existing checkpoint should be possible

### Query
```python
us = best_model.get_latent(X)  # latent values from data (train, val, sampled_data)
best_model.sample(do=None, us=None)
```

- Modes for `sample(...)`:
  - `us=None, do=None` → Observational data
  - `us=None, do=(None, 42, None)` → Interventional data
  - `us=us, do=(None, 42, None)` → Counterfactual data
  - `us` can be partially `None`; missing latents are sampled (e.g., `us=(None, None, u3)`).
  - If `us=(u1, u2, u3)`, counterfactuals are deterministic.
