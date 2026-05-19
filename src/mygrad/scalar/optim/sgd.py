class SGD:
    def __init__(self, params, lr=0.001):
        self.p = params
        self.lr = lr

    def step(self):
        for p in self.p:
            p.data -= self.lr * p.grad
