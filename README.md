[![tox](https://github.com/directgroup/direct/actions/workflows/tox.yml/badge.svg)](https://github.com/directgroup/direct/actions/workflows/tox.yml)
[![pylint](https://github.com/directgroup/direct/actions/workflows/pylint.yml/badge.svg)](https://github.com/directgroup/direct/actions/workflows/pylint.yml)
[![black](https://github.com/directgroup/direct/actions/workflows/black.yml/badge.svg)](https://github.com/directgroup/direct/actions/workflows/black.yml)

# DIRECT
DIRECT is the Deep Image REConstruction Toolkit that implements state-of-the-art inverse problem solvers. It stores
inverse problem solvers such as the Learned Primal Dual algorithm and Recurrent Inference Machine, which were part of the winning solution in Facebook & NYUs FastMRI challenge in 2019 and the Calgary-Campinas MRI reconstruction challenge at MIDL 2020.

<div align="center">
  <img src=".github/direct.png"/>
</div>

## Installation
See [install.md](install.md).

## Quick Start
See [getting_started.md](getting_started.md), check out the [documentation](https://docs.aiforoncology.nl/direct).
In the [projects](projects) folder examples are given on how to train models on public datasets.

## Baselines and trained models.
We provide a set of baseline results and trained models in the [DIRECT Model Zoo](model_zoo.md).

## License
DIRECT is released under the [Apache 2.0 License](LICENSE).

## Citing DIRECT
If you use DIRECT in your own research, or want to refer to baseline results published in the
 [DIRECT Model Zoo](model_zoo.md), please use the following BiBTeX entry:

```BibTeX
@misc{DIRECTTOOLKIT,
  author =       {Teuwen, Jonas and Yiasemis, George and Moriakov, Nikita and Karkalousos, Dimitrios  and Caan, Matthan},
  title =        {DIRECT: Deep Image REConstruction Toolkit},
  howpublished = {\url{https://github.com/directgroup/direct}},
  year =         {2021}
}
```
