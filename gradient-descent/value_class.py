class Value:
    def __init__(self, data, _children=(), _ops=""):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _ops
        self._backward = lambda: None

    def __add__(self, other):
        addition = self.data + other.data
        out = Value(addition, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        multiplication = self.data * other.data
        out = Value(multiplication, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad = self.data * out.grad

        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data})"

    # depth first search for topoligical list
    def backwards(self):
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


a = Value(2.0)
b = Value(-4.0)
c = a * b
d = c + a

d.backwards()
print(f"a: {a.grad}, b: {b.grad}, c: {c.grad}")
