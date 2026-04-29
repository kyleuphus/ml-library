class Module:
    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        return []
