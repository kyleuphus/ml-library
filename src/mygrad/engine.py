import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        addition = self.data + other.data
        out = Value(addition, (self, other))

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        # passes correct backward function to out
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
        out = Value(multiplication, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        power = self.data ** (other)
        out = Value(power, (self,), f"**{other}")

        def _backward():
            local_derivative = other * self.data ** (other - 1)
            self.grad += local_derivative * out.grad

        out._backward = _backward

        return out

    def __abs__(self):

        def _backward():
            if self.data > 0:
                local_derivative = 1
            elif self.data == 0:
                local_derivative = 0
            else:
                local_derivative = -1

            self.grad += local_derivative * out.grad

        absolute = abs(self.data)
        out = Value(absolute, (self,), f"|{self}|")

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1.0) / (math.exp(2 * x) + 1.0)

        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    # depth first search for topological list
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
