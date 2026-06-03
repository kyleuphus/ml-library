import numpy as np


class FakeTensor:
    def __init__(self, grad):
        self.grad = grad

    def unbroadcast(self, target_shape):
        if len(self.grad.shape) == len(target_shape):
            axes = []
            index = 0
            while index < len(self.grad.shape):
                if self.grad.shape[index] == target_shape[index]:
                    index += 1
                else:
                    axes.append(index)
                    index += 1
            self.grad = self.grad.sum(axis=tuple(axes), keepdims=True)
        else:
            difference = len(self.grad.shape) - len(target_shape)
            axes = list(range(difference))
            self.grad = self.grad.sum(axis=tuple(axes))


t = FakeTensor(np.ones((2, 3)))
t.unbroadcast((1, 3))
print(t.grad.shape)  # should be (1, 3)

t = FakeTensor(np.ones((4, 2, 3)))
t.unbroadcast((3,))
print(t.grad.shape)  # should be (3,)

t = FakeTensor(np.ones((3,)))
t.unbroadcast((3,))
print(t.grad.shape)  # should be (3,)
