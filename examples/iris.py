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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.mygrad.nn.losses import mse, softmax_cross_entropy
from src.mygrad.nn.activations import relu 
from src.mygrad.nn.mlp import MLP
from src.mygrad.engine import Tensor
from src.mygrad.training.loop import train 

# %%
# dataset import & loading
iris = load_iris()
X = iris["data"]
y = iris["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
y_train = np.eye(3)[y_train]
y_test = np.eye(3)[y_test]

# %% 
# loading into tensors
X_train = Tensor(X_train)
X_test = Tensor(X_test)
y_train = Tensor(y_train)
y_test = Tensor(y_test)

# %%
# training the MLP
mlp = MLP(4, [8, 8], 3, activation=relu)
train(mlp, X_train, y_train, softmax_cross_entropy, lr=.015, lr_decay=1.0, epochs=100000)

# %%
# testing MLP with updated params
pred = np.argmax(mlp(X_test).data, axis=1)
true = np.argmax(y_test.data, axis=1)
np.equal(pred, true)
