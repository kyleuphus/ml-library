from mygrad.engine import Tensor
import numpy as np
import math


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
        # xavier initialization
        normalized = math.sqrt(6) / (math.sqrt(in_features + out_features))
        self.weight = Tensor(
            rng.uniform(-normalized, normalized, (in_features, out_features))
        )
        self.bias = Tensor(rng.uniform(-1 / in_features, 1 / in_features, out_features))

    def parameters(self):
        return [self.weight, self.bias]

    def forward(self, x: Tensor):
        return x @ self.weight + self.bias


class MLP(Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: list,
        out_features: int,
        activation: callable,
    ):
        dimensions = [in_features, *hidden_features, out_features]
        sizes = zip(dimensions, dimensions[1:])
        self.activation = activation
        self.layers = []
        global rng
        rng = np.random.default_rng(42)
        for pair in sizes:
            self.layers.append(Linear(pair[0], pair[1]))

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x):
        i = 1
        activation = self.activation
        layers = len(self.layers)
        for layer in self.layers:
            x = layer(x)
            if i != layers:
                x = activation(x)
            i += 1
        return x
