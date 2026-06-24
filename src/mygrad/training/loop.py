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
    input: np.ndarray,
    target: np.ndarray,
    loss: callable,
    lr: float = 0.0001,
    lr_decay: float = 1.0,
    epochs: int = 1000,
    batch_size: int = 64,
):
    grad = SGD(mlp.parameters(), lr=lr)
    initial_lr = lr
    for epoch in range(epochs):
        x, y = shuffle(input, target)
        batches = []
        for i in range(0, len(input), batch_size):
            pair = (Tensor(x[i : i + batch_size]), Tensor(y[i : i + batch_size]))
            batches.append(pair)
        for batch in batches:
            _loss = _step(mlp, grad, loss, x=batch[0], y=batch[1])
        grad.lr = initial_lr * math.pow(lr_decay, epoch)
        print(f"epoch: {epoch}, loss: {_loss}")


def _step(mlp: MLP, grad: SGD, loss: callable, x: Tensor, y: Tensor):
    mlp.zero_grad()
    out = mlp(x)
    _loss = loss(out, y)
    _loss.grad = np.ones_like(_loss.data)
    _loss.backward()
    grad.step()
    return _loss.data


def shuffle(x: np.ndarray, y: np.ndarray):
    shuffled_indices = np.random.permutation(len(x))
    shuffled_x = x[shuffled_indices]
    shuffled_y = y[shuffled_indices]
    return shuffled_x, shuffled_y
