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
from src.mygrad.training.loop import train
from src.mygrad.nn.activations import relu 
from src.mygrad.nn.losses import categorical_cross_entropy

# %%
mlp = MLP(4, [8, 8], 2, relu)
input = Tensor(np.ones((32, 4)))
target = Tensor(np.array([6.0, 6.0]))

# %%
train(mlp, input, target, categorical_cross_entropy)


