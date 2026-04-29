import random
from mygrad.engine import Value
from mygrad.nn.module import Module


class Neuron(Module):
    def __init__(self, nin, activation=None):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act = activation.lower() if activation else None

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        # weight * input for respective index in zipped tuple, adding to bias
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.act == "tanh":
            return activation.tanh()
        elif self.act == "relu":
            return activation.relu()
        elif self.act in [None, "linear"]:
            return activation


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs


class MLP(Module):
    def __init__(self, nin, nouts):
        size = [nin] + nouts
        self.layers = [Layer(size[i], size[i + 1]) for i in range(len(nouts))]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
