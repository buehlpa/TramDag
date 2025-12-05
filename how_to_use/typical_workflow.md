
# Typical Workflow for a new tramdag experiment

This can be used in a .ipynb notebook following the steps below

If you already have a setup experiments and / or a trained model. you can proceed also just with the steps 4. and 5. 

The notebooks:
- example_notebooks/1_create_configuration_file.ipynb
- example_notebooks/2_fit_tramdag_model.ipynb
show how a configuration can be created and in another file trained based on the saved configuration


### 1. Data
- simulate data or use observational data in a pandas dataframe e.g.

```
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000,n_features=3,centers=4,cluster_std=1.2,random_state=42)
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["label"] = y
```

- train val test split  e.g.

```
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
```


### 2. Model Configuration
Tramdag Configuration refers to configuring the DAG structure aswell as the neural network terms to be used to learn the strucutral functions

- experimentnames / paths

```
from tramdag import TramDagConfig

cfg=TramDagConfig()
cfg.setup_configuration(EXPERIMENT_DIR='experiment_1') # works also without arguments, then default paths are used
```

- datatypes

see [how to set datatypes](/how_to_use/Ordinal_Continous_vars.md)
```
data_type= {'x1':'continous',
            'x2':'continous', 
            'x3':'continous'} 

cfg.set_data_type(data_type)
```

- Adjacency matrix 

```
cfg.set_meta_adj_matrix(seed=123) #Create the (Meta) Adjacency Matrix
```
<img width="1637" height="997" alt="example _Xn_Yo" src="https://github.com/buehlpa/TramDag/blob/main/docs/images/adj_matrix.png" />



- Neural Networks 


```
cfg.set_tramdag_nn_models() 
```
<img width="200" height="100" alt="example _Xn_Yo" src="https://github.com/buehlpa/TramDag/blob/main/docs/images/nn_models.png" />



if you have ordinal nodes in the your dataset the configuration needs also to get the levels of these nodes.
You can either provide them by manually write them to the configuration.json file or calculated it from data with:

```
cfg.compute_levels(train_df)
```






### 3. Modelfit

There are 3 options to load the model from the configuration file 

1. use the cfg object either directly if created in the same notebook:

or load it from the existing json with:

```
cfg=TramDagModel.load_json(CONF_DICT_PATH="experiment_1/configuration.json")
```

Then load the model directly from the config file use the flag `set_initial_weights = True` and provide `initial_data = train_df` if you 
want to start an R subprocess for the weight initialization 

```
from tramdag import TramDagModel
device='cpu'
td_model = TramDagModel.from_config(cfg, set_initial_weights=False,verbose=True,debug=False,device=device,initial_data = train_df) 
```

You can also load the model directly from a directory if you already trained a model

```
from tramdag import TramDagModel
device='cpu'
td_model =TramDagModel.from_directory(EXPERIMENT_DIR)
```

Additionaly:

to fit a model in the background adjust a python script and run it in  a screen in the background e.g.

[2_fit_tramdag_in_background.py](/example_notebooks/2_fit_tramdag_in_background.py)
- 

### 4. Fit diagnostics

- loss

with the argument `variable` you can choose a single variable or just leave it empty to plot all in the same graph

```
td_model.plot_loss_history()
```

- linear shifts

if you already know the theoretical shifts you can plot them directly as dotted lines with the `ref_lines` argument

```
td_model.plot_linear_shift_history(ref_lines={'x2':[1.973827],'x3':[-0.1815344, -1.0012274 ]})
```


- intercepts

```
td_model.plot_simple_intercepts_history(ref_lines={'x3':[-1.998953,  0.426397,  1.032376]})
```


- latents

```
td_model.plot_latents(train_df)
```

- final nll

```
td_model.nll(train_df)
```

- transformation functions

plot for a dataframe the transformation functions `plot_n_rows`
```
td_model.plot_hdag(train_df,variables=['x1','x2','x3'],plot_n_rows=1)
```

- model summary

```
td_model.summary()
```


### 5. Sampling

see [how to sample](/how_to_use/sampling.md)

- Observational

```
samples_observational, latents = td_model.sample()
td_model.plot_samples_vs_true(test_df,samples_observational)
```

- Interventions

```
samples_inter, latents = td_model.sample(do_interventions={'x1':-3.0})
inter_df=dgp(n_obs=10_000, doX=[-3,None , None])
td_model.plot_samples_vs_true(inter_df,samples_inter)
```

- Counterfactuals


```
subset = train_df.iloc[:100]
subset =pd.DataFrame(subset)
subset.info()

u_df=td_model.get_latent(df = subset) # return of df should be x3_U_lower X3_U_upper

rsamples, latents = td_model.sample(predefined_latent_samples_df=u_df)

td_model.plot_samples_vs_true(subset,rsamples)
```
