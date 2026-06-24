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
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.mygrad.nn.losses import softmax_cross_entropy
from src.mygrad.nn.activations import relu
from src.mygrad.nn.mlp import MLP
from src.mygrad.engine import Tensor
from src.mygrad.training.loop import train

# %%
# dataset import & loading
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
X_train, X_test = X_train.astype(int), X_test.astype(int)
y_train, y_test = y_train.astype(int), y_test.astype(int)
y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]
X_test = Tensor(X_test)
y_test = Tensor(y_test)
# %% 
# creating mlp
mlp = MLP(784, [256, 128], 10, activation=relu)
train(mlp, X_train, y_train, softmax_cross_entropy, lr=.01, epochs=30, batch_size=64)

