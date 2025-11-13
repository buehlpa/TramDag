#   **TRA**nsformation **M**odels in **D**irected **A**cyclic **G**raphs (**TRAMDAG**)
A pytorch implemetation of Interpretable Neural Causal Models with TRAM-DAGs (https://arxiv.org/abs/2503.16206)

This framework uses ordinal and continous transformation models to learn the functions on the edges on a directed acyclic graph.

The according implentations for the ordinal tram and the continous tram are in the the github repos:

[ontram](github.com/liherz/ontram_pytorch) and [contram](github.com/buehlpa/contram_pytorch)



![tramdag](https://github.com/buehlpa/TramDag/blob/main/doc/images/tramdag.png)



## Installation



It is recommended to use a virtual environment such as anaconda env like: 
on https://www.anaconda.com/docs/getting-started/anaconda/install

```bash
  conda create -n tramdag python=3.9
  conda activate tramdag
```

the **tramdag** package is currently on a testserver, install with:
```bash
  pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple tramdag
```

**optional**: If the weights of the Simple Intercepts should be initialized not randomly one can use a warmstart via R subprocess which needs an installation of R and according packages:

- install R : https://cran.r-project.org/
- install packages in R:
 
```bash
install.packages(c("tram", "ordinal", "readr", "MASS"))
```
## Documentation
under /doc
## Example Notebooks

under /example_notebooks some examples are displayed

Reproducing simulation experiments from [TRAMDAG paper](https://arxiv.org/abs/2503.16206)

- [6.1 Continous_complex DGP complex Shift](example_notebooks/Continous_3_vars_complex_DGP_CS.ipynb)
- [6.1 Continous_linear DGP complex Shift](example_notebooks/Continous_3_vars_linear_DGP_CS.ipynb)
- [6.1 Continous_linear DGP linear Shift](example_notebooks/Continous_3_vars_linear_DGP_LS.ipynb)
- [6.2 Mixed_linear DGP complex Shift](example_notebooks/Mixed_3_vars_linear_DGP_LS.ipynb)



## Limitations


- Maximum input per **node** for  Complex Shift or Complex intercept is **9**
- Currently only tabular data is supported , but potentially the framework can be extended for to other modalities e.g. images via torch loader.
## Authors

- [@buehlpa](https://www.github.com/buehlpa)
- [@oduerr](https://www.github.com/oduerr)
- [@bsick](https://www.github.com/bsick)


## Citation
If you intend to use this repository for your experiments please cite:


```
@misc{sick2025interpretableneuralcausalmodels,
  title={Interpretable Neural Causal Models with TRAM-DAGs},
  author={Beate Sick and Oliver Dürr},
  year={2025},
  eprint={2503.16206},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2503.16206}
}
```


