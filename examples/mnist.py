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
from mygrad.nn.losses import softmax_cross_entropy
from mygrad.nn.activations import relu
from mygrad.nn.mlp import MLP
from mygrad.engine import Tensor
from mygrad.training.loop import train

# %%
# dataset import & loading
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)
X_train, X_test = X_train.astype(float), X_test.astype(float)
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = y_train.astype(int), y_test.astype(int)
y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test]
X_test = Tensor(X_test)
y_test = Tensor(y_test)

# %%
# creating mlp
mlp = MLP(784, [256, 128], 10, activation=relu)
train(mlp, X_train, y_train, softmax_cross_entropy, lr=0.01, epochs=200, batch_size=64)

# %%

pred = np.argmax(mlp(X_test).data, axis=1)
true = np.argmax(y_test.data, axis=1)
print(f"accuracy = {(pred == true).mean() * 100}%")
