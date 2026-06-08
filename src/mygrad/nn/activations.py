from mygrad.engine import Tensor
import numpy as np


def relu(x: Tensor):
    _relu = np.maximum(0, x.data)
    out = Tensor(_relu, (x,), "relu")

    def _backward():
        x.grad += (x.data > 0) * out.grad

    out._backward = _backward

    return out


def tanh(x: Tensor):
    _tanh = (np.exp(2.0 * x.data) - np.array([1.0])) / (
        np.exp(2.0 * x.data) + np.array([1.0])
    )

    out = Tensor(_tanh, (x,), "tanh")

    def _backward():
        x.grad += (1 - np.power(out.data, 2.0)) * out.grad

    out._backward = _backward
    return out


def sigmoid(x: Tensor):
    _sigmoid = (np.array([1.0])) / (np.array([1.0]) + np.exp(-x.data))

    out = Tensor(_sigmoid, (x,), "sigmoid")

    def _backward():
        x.grad += (out.data * (1 - out.data)) * out.grad


def leaky_relu(x: Tensor, alpha=0.01):
    lrelu = np.where(x.data < 0, x.data * alpha, x.data)
    out = Tensor(lrelu, (x,), "leaky_relu")

    def _backward():
        x.grad += np.where(x.data < 0, alpha, 1) * out.grad

    out._backward = _backward

    return out


def softmax(x: Tensor):
    exp = np.exp(x.data)
    exp_sum = exp.sum()
    _softmax = exp / exp_sum
    out = Tensor(_softmax, (x,), "softmax")

    def _backward():
        x.grad += (np.diag(_softmax) - np.outer(_softmax, _softmax)) @ out.grad

    out._backward = _backward

    return out
