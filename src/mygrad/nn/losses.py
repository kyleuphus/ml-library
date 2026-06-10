import numpy as np
from mygrad.engine import Tensor
from src.mygrad.nn.activations import softmax


def mse(out: Tensor, target: Tensor):
    loss = ((out - target) * (out - target)).sum() / float(out.data.size)
    return loss


def binary_cross_entropy(out: Tensor, target: Tensor):
    loss = -(
        target * out.log()
        + (Tensor(np.array([1.0])) - target) * (Tensor(np.array([1.0])) - out).log()
    )
    return loss


def categorical_cross_entropy(out: Tensor, target: Tensor):
    loss = -((target * out.log()).sum())
    return loss


def softmax_cross_entropy(out: Tensor, target: Tensor):
    s = softmax(out)
    loss = -(np.sum(target.data * s.log().data))
    result = Tensor(loss, (out, target), "softmax_cross_entropy")

    def _backward():
        out.grad += (s.data - target.data) / out.data.shape[0]

    result._backward = _backward
    return result
