from mygrad.nn import MLP
from mygrad.optim import SGD

xs = [
    [2.0, 3.0],
    [-1.0, 1.0],
    [0.5, -2.0],
    [1.0, 1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(2, [12, 12, 1], "tanh")

step = 0.005

grad = SGD(model.parameters(), 0.001)

for epoch in range(1000):
    # forward pass
    ypred = [model(x)[0] for x in xs]

    loss = sum(((pred - actual) ** 2 for pred, actual in zip(ypred, ys))) / len(ys)

    model.zero_grad()

    loss.backward()

    grad.step()

    print(f"Epoch: {epoch}, Loss: {loss.data}")

print(ypred)
