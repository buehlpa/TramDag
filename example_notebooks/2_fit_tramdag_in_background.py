from tramdag import TramDagModel,TramDagConfig
from sklearn.model_selection import train_test_split
import pandas as pd



device='cpu'
epochs = 2000

df=pd.read_csv("mydata.csv")
# data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


cfg = TramDagConfig.load_json('experiment_1/configuration.json')
cfg.compute_levels(df=train_df)

td_model = TramDagModel.from_config(cfg, set_initial_weights=True,verbose=True,debug=False,device=device,initial_data =train_df )


td_model.fit(train_df, val_df,
            #  train_list=['bp'], #Training only on a subset of the model
             learning_rate=1e-2,
             epochs=epochs,batch_size=10_00,
             verbose=False,debug=False,
             device=device,
             num_workers = 8,
             persistent_workers = True,
             prefetch_factor = 8,       #For DataLoader
             train_mode = "sequential") #Parallel is better for many nodes 