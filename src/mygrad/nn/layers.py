import random
from mygrad.engine import Value


class Neuron:
    def __init__(self, nin, activation=None):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act = activation.lower() if activation else None

    def __call__(self, x):
        # weight * input for respective index in zipped tuple, adding to bias
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.act == "tanh":
            return activation.tanh()
        elif self.act == "relu":
            return activation.relu()
        elif self.act in [None, "linear"]:
            return activation


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in nout]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs


class MLP:
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
