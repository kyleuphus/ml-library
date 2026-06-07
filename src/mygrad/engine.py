import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = np.array(data)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __matmul__(self, other):
        matmul = np.matmul(self.data, other.data)
        out = Tensor(matmul, (self, other), "matmul")

        def _backward():
            if out.grad.shape == ():
                self.grad += out.grad * other.data
                other.grad += self.data * out.grad
            elif out.grad.ndim == 1 and other.data.ndim == 1:
                self.grad += np.outer(out.grad, other.data)
                other.grad += self.data @ out.grad
            else:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad

        out._backward = _backward

        return out

    def __add__(self, other):
        add = self.data + other.data
        out = Tensor(add, (self, other), "add")

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward

        return out

    def __mul__(self, other):
        mul = self.data * other.data
        out = Tensor(mul, (self, other), "mul")

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.grad.shape)
            other.grad += unbroadcast(self.data * out.grad, other.grad.shape)

        out._backward = _backward

        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        div = self.data / other.data
        out = Tensor(div, (self, other), "div")

        def _backward():
            self.grad += 1 / other.data * out.grad
            other.grad += -self.data / (other.data * other.data) * out.grad

        out._backward = _backward

        return out

    def sum(self):
        sum = self.data.sum()
        out = Tensor(sum, (self,), "sum")

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward

        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        neg = -self.data
        out = Tensor(neg, (self,), "neg")

        def _backward():
            self.grad += -1 * out.grad

        out._backward = _backward

        return out

    def relu(self):
        relu = np.maximum(0, self.data)
        out = Tensor(relu, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def _back(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    _back(child)
                topo.append(v)

        _back(self)

        for node in reversed(topo):
            node._backward()


def unbroadcast(grad, target_shape):
    if len(grad.shape) == len(target_shape):
        axes = []
        index = 0
        while index < len(grad.shape):
            if grad.shape[index] == target_shape[index]:
                index += 1
            else:
                axes.append(index)
                index += 1
        return grad.sum(axis=tuple(axes), keepdims=True)
    else:
        difference = len(grad.shape) - len(target_shape)
        axes = list(range(difference))
        return grad.sum(axis=tuple(axes))
