# Tie-Decay Epidemics

[![PyPI Version](https://img.shields.io/pypi/v/tie-decay-epidemics.svg)](https://pypi.org/project/tie-decay-epidemics/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/tie-decay-epidemics.svg)](https://pypi.org/project/tie-decay-epidemics/)

Tie-Decay Epidemics contains the scripts that allow you to simulate epidemic spreading on a tie-decay network.

---

## Installation

To install Tie-Decay Epidemics, run this command in your terminal:

```bash
$ pip install -U tie-decay-epidemics
```

This is the preferred method to install Tie-Decay Epidemics, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Quick Start
First, import all the basic packages.
```python
>>> import tie_decay_epidemics as tde
>>> import numpy as np
>>> import pandas as pd
```

Construct a tie-decay network.
```python
>>> nodes = np.arange(10)
>>> G = tde.TieDecay_Graph(nodes)
```

Simulate an SIS process on a tie-decay network.
```python
>>> nodes = np.arange(10)
>>> infected = np.array([0])
>>> edgelist = pd.read_csv("FILE_NAME.csv")
>>> rateSI = 0.2
>>> rateIS = 0.1 
>>> alpha = 0.01
>>> SIS = tde.TieDecay_SIS(nodes, infected, edgelist, rateSI, rateIS, alpha)
>>> SIS.run(max_time = 1000)
```

## Citing
If you use our work in an academic setting, please cite our code.
