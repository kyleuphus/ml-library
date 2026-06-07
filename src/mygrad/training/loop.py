# initialize random weights + biases
# forward pass
# calculate loss
# zero gradient and set loss.grad = 1
# loss.backward()
# for param in params, param += lr * (-param.grad) (sgd)
# forward pass and repeat from there
import numpy as np

from mygrad.nn.mlp import Tensor, MLP
from mygrad.optim.sgd import SGD


def train(
    mlp: MLP, input: Tensor, target: Tensor, loss: str = "mse", epochs: int = 1000
):
    grad = SGD(mlp.parameters())
    i = 0
    for epoch in range(epochs):
        out = mlp(input)
        mse = ((out - target) * (out - target)).sum() / float(out.data.size)
        if i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            print(mse.data)
        mlp.zero_grad()
        mse.grad = np.ones_like(mse.data)
        mse.backward()
        grad.step()
        i += 1
