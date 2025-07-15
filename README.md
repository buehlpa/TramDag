Pytorch implementation of **TRA**nsformation **M**odels in **D**irected **A**cyclic **G**raphs (**TRAMDAG**)

🔧👷 Work in progress 

The Repos for the separate continous case and the ordinal case are in 

- github.com/buehlpa/contram_pytorch
- github.com/liherz/ontram_pytorch


This repo allows to independently model each node in a graph and model the SCM functions
such that if the SCM is learned, interventions can be applied and Interventional distributions can be obtained


In addition to the original implenetation in R this repo allows for:

-  interactions in Complex models
  
- independent training of NNs instead of MAF.


Original Paper with R Code: https://arxiv.org/abs/2503.16206


Reproduced experiments from the orignal paper §Section 6.1 under /reproducing_tramdag_experiments




![image](https://github.com/user-attachments/assets/c3396efd-bc30-4e0b-a947-d1908a3285d0)
