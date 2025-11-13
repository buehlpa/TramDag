

## Encoding of Ordinal and Continuous Variables in TramDAG

In the **TramDAG framework**, nodes can act both as **predictors (X)** and as **targets (Y)**.  
The way a node is modelled depends on its data type:

- **Continuous variables** (`∈ ℝ`)  
  Always modelled the same way for neural networks (regression-style output).

- **Ordinal variables**  
  Can be modelled in different ways:
  - As **continuous inputs** (e.g. standardized integers, or raw integers from the dataset).
  - As **nominal/categorical inputs** (e.g. one-hot encoding).
---

### Example: Models for X4 and X5

The figure below shows an example where nodes `X1`–`X4` are **ordinal**, and `X5` is **continuous**:

- **Model X4**  
  - Inputs:  
    - `X2` is treated as continuous (40 classes).  
    - `X1` and `X3` are one-hot encoded.  
  - Target:  
    - `X4` is ordinal → trained with an ordinal loss.  

- **Model X5**  
  - Input:  
    - `X4` is treated as continuous.  
  - Target:  
    - `X5` is continuous → trained with a continuous loss.  

---

### Key Idea
- The **X-specification** controls how variables are **encoded as predictors**.  
- The **Y-specification** controls how variables are **encoded as targets** (i.e. which loss is applied).

---

<img width="1637" height="997" alt="example _Xn_Yo" src="https://github.com/user-attachments/assets/93fada9c-cfcd-4507-b214-7f657c9b906d" />




![image](https://github.com/user-attachments/assets/c3396efd-bc30-4e0b-a947-d1908a3285d0)
