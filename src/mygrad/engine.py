import math


class Value:
    def __init__(self, data, _children=(), _ops=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _ops
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        addition = self.data + other.data
        out = Value(addition, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        multiplication = self.data * other.data
        out = Value(multiplication, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1.0) / (math.exp(2 * x) + 1.0)

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    # depth first search for topoligical list
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
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()


a = Value(5)
print(2 - a)
