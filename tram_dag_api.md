
Input np/pandas data frames df_train, df_val

# 1) Load or create config (typed; no I/O unless you call .save)
cfg = load_config("config.yaml")  # or TramConfig(**dict)

# 2) Build dataset explicitly (no global state; no writes)
train = build_from_dataframe(df_train, cfg)
val   = build_from_dataframe(df_val,  cfg)

# 3) Build the TRAM-DAG model from configuration
model = model_from_config(cfg)          # holds all node specs


# 4) Train with an explicit trainer; returns history and best model
trainer = Trainer(epochs=100, lr=1e-2, scheduler="cosine", device="auto")
best_model, history = trainer.fit(model, train,val, callbacks=[])
# optional: Logging the Traininprocess (e.g. with W&B ), EarlyStopping, Checkpoint
# Note that warmstarting with given shift / intercept terms should be possible

# 5) Inspect/interpret 
print(best_model.summary())                   # parameters, structure
print(best_model.get_shift_weights())         # linear shift coefficients if applicable

# 6) Persist explicitly 
save_model(bestmodel, "runs/best_model.pt")      # no auto-write
save_history(history, "runs/history.json")
save_cfg(cfg, "runs/config.yaml")

# Loading / Retraining
model = load_model("runs/best_model.pt")
history = load_history(history, "runs/history.json") 
cfg = load_cfg(cfg, "runs/history.json")

#  Note that retraining from existing checkpoint should be possible

# Querry 
us = get_latent(best_model, X) 
#latent values from Data (train, val, sampled_data)

sample(best_model, do=None, us=None) 
# return data 
#      us=None, do=None → Obersvational Data
#      us=None, do=(Na,42,Na) → Interventional Data
#      us=us , do=(Na,42,Na) → Counterfactual Data
#      Note that us can be partly None in that case samples are chosen
#           Example us = (Na, Na, u3) 
#      Note that if us = (u1, u2, u3) we get deterministic CF
