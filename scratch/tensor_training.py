# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from src.mygrad.nn.mlp import Tensor, MLP
from src.mygrad.optim.sgd import SGD
from src.mygrad.training.loop import train

# %%
mlp = MLP(4, [8, 8], 2)
input = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
target = Tensor(np.array([6.0, 6.0]))

# %%
train(mlp, input, target)


