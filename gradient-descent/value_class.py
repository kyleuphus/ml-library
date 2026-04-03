class Value:
    def __init__(self, data, _children=(), _ops=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _ops

    def __add__(self, other):
        addition = self.data + other.data
        return Value(addition, (self, other), "+")

    def __mul__(self, other):
        multiplication = self.data * other.data
        return Value(multiplication, (self, other), "*")

    def __repr__(self):
        return f"Value(data={self.data})"


a = Value(8)
b = Value(12)
c = a * b
c += b
print(c._prev)
