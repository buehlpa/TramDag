

## Encoding of Ordinal and Continuous Variables in TramDAG
This Readme shows how the settings for the datatype in tramdag are internally treated

e.g. :

````
data_type= {'x1':'ordinal_Xn_Yo',
            'x2':'ordinal_Xc_Yo',
            'x3':'ordinal_Xn_Yo',
            'x4':'ordinal_Xc_Yo',
            'x5':'continous'} 

cfg.set_data_type(data_type)
````



In the **TramDAG framework**, nodes can act both as **predictors (X)** and as **targets (Y)**.  
The way a node is modelled depends on its data type:

- **Continuous variables** (`∈ ℝ`)  
  Always modelled the same way for neural networks (regression-style output).

- **Ordinal variables**  
  Can be modelled in different ways:
  - As **continuous inputs** (e.g. standardized integers, or raw integers from the dataset).
  - As **nominal/categorical inputs** (e.g. one-hot encoding) (handled model internally, controlled by data_type).

### Key Idea
- The **X-specification** controls how variables are **encoded as predictors**.  
- The **Y-specification** controls how variables are **encoded as targets** (i.e. which loss is applied).

There are 4 ways to set the ordinal specifications :

````
ordinal_Xn_Yo   
ordinal_Xn_Yc
ordinal_Xc_Yo
ordinal_Xc_Yc  : this is the same as modelling the variable as continous
````




### Example: Models for X4 and X5

The figure below shows an example where nodes `X1`–`X4` are **ordinal**, and `X5` is **continuous**:


the example only shows the models for X4 and X5 , the others are omitted 

- **Model X4**  
  - Inputs:  
    - `X2` is treated as continuous (40 classes).  
    - `X1` and `X3` are one-hot encoded automatically since Yo is set
  - Target:  
    - `X4` is ordinal → trained with an ordinal loss.  

- **Model X5**  
  - Input:  
    - `X4` is treated as continuous.  
  - Target:  
    - `X5` is continuous → trained with a continuous loss.  

---








---

<img width="1637" height="997" alt="example _Xn_Yo" src="https://github.com/buehlpa/TramDag/blob/main/doc/images/example _Xn_Yo.png" />



