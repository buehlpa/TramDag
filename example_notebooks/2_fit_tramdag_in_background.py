"""Fit a TRAM-DAG from a script, e.g. inside `screen` / `nohup` or on a cluster node.

The JAX engine trains all nodes jointly and uses a GPU automatically if JAX finds one.
Training resumes from the stored `last_params.npz` / histories if the script is restarted
with a larger number of epochs.
"""
from sklearn.model_selection import train_test_split
import pandas as pd

from tramdag import TramDagConfig, TramDagModel

device = "auto"   # "cpu" to force the CPU
epochs = 2000

df = pd.read_csv("mydata.csv")
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

cfg = TramDagConfig.load_json("experiment_1/configuration.json")
cfg.compute_levels(df=train_df)

# set_initial_weights=True warm-starts intercepts / linear shifts with R (tram::Colr, MASS::polr)
td_model = TramDagModel.from_config(cfg, set_initial_weights=False, verbose=True, device=device, initial_data=train_df)

td_model.fit(
    train_df,
    val_df,
    # train_list=['bp'],   # train only a subset of the nodes
    learning_rate=1e-2,
    epochs=epochs,
    batch_size=1000,
    verbose=True,
)
print(td_model.nll(test_df))
