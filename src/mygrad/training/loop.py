# initialize random weights + biases
# forward pass
# calculate loss
# zero gradient and set loss.grad = 1
# loss.backward()
# for param in params, param += lr * (-param.grad) (sgd)
# forward pass and repeat from there
import numpy as np
import math
from mygrad.nn.mlp import Tensor, MLP
from mygrad.optim.sgd import SGD


def train(
    mlp: MLP,
    input: Tensor,
    target: Tensor,
    loss: callable,
    lr: float = 0.0001,
    lr_decay: float = 1.0,
    epochs: int = 1000,
):
    grad = SGD(mlp.parameters(), lr=lr)
    initial_lr = lr
    for epoch in range(epochs):
        out = mlp(input)
        _loss = loss(out, target)
        grad.lr = initial_lr * math.pow(lr_decay, epoch)
        if (epoch + 1) % 500 == 0:
            print(f"epoch: {epoch + 1} : {_loss.data}")
        mlp.zero_grad()
        _loss.grad = np.ones_like(_loss.data)
        _loss.backward()
        grad.step()
