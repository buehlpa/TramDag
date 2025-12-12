#   TRA*nsformation* M*odels* in D*irected* A*cyclic* G*raphs* (TRAMDAG)
A pytorch implemetation of Interpretable Neural Causal Models with TRAM-DAGs (https://arxiv.org/abs/2503.16206)

The framework uses ordinal and continuous transformation models to learn the intercepts and linear or nonlinear shift terms that define the conditional transformation functions on each node, thereby modeling how each parent variable causally shifts the distribution of its child in the DAG.

The according implementations for the *ordinal tram* and the *continous tram* are in the the github repos:

[ontram](https://github.com/liherz/ontram_pytorch) and [contram](https://github.com/buehlpa/contram_pytorch)



![tramdag](https://github.com/buehlpa/TramDag/blob/main/docs/images/tramdag.png)



## Installation

It is recommended to use a virtual environment such as anaconda env like: 
https://www.anaconda.com/docs/getting-started/anaconda/install

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
Read the guide below for usage instructions
- [HOW-TO](how_to_use/typical_workflow.md)


Full API CODE documentation
- [Code Documentation](https://buehlpa.github.io/TramDag/tramdag.html)



## Example Notebooks

under /example_notebooks some examples are displayed

Reproducing simulation experiments from [TRAMDAG paper](https://arxiv.org/abs/2503.16206)


- [6.1 Continous_complex DGP complex Shift (L1 L2)](example_notebooks/Continous_3_vars_complex_DGP_CS.ipynb)[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buehlpa/tramdag/blob/main/example_notebooks/Continous_3_vars_complex_DGP_CS.ipynb)
- [6.1 Continous_linear DGP complex Shift  (L1 L2)](example_notebooks/Continous_3_vars_linear_DGP_CS.ipynb)[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buehlpa/tramdag/blob/main/example_notebooks/Continous_3_vars_linear_DGP_CS.ipynb)
- [6.1 Continous_linear DGP linear Shift   (L1 L2)](example_notebooks/Continous_3_vars_linear_DGP_LS.ipynb)[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buehlpa/tramdag/blob/main/example_notebooks/Continous_3_vars_linear_DGP_LS.ipynb)
- [6.2 Mixed_linear DGP complex Shift      (L1 L2 L3)](example_notebooks/Mixed_3_vars_linear_DGP_LS.ipynb)[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/buehlpa/tramdag/blob/main/example_notebooks/Mixed_3_vars_linear_DGP_LS.ipynb)

if you are using the notebook in colab install tramdag and the r packages with:

```bash
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple tramdag
!R -e "install.packages(c('tram','ordinal'),repos = c(ETH = 'https://stat.ethz.ch/CRAN',CRAN = 'https://cloud.r-project.org'))"
```

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


