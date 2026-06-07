from mygrad.engine import Tensor
import numpy as np

rng = np.random.default_rng()


class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = np.zeros_like(param.data)

    def parameters(self):
        return []

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        self.weight = Tensor(rng.random(size=(in_features, out_features)))
        self.bias = Tensor(rng.random(size=(out_features,)))

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x: Tensor):
        return x @ self.weight + self.bias


class MLP(Module):
    def __init__(self, in_features: int, hidden_features: list, out_features: int):
        dimensions = [in_features, *hidden_features, out_features]
        sizes = zip(dimensions, dimensions[1:])
        self.layers = []
        for pair in sizes:
            self.layers.append(Linear(pair[0], pair[1]))

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
