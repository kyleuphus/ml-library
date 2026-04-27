from mygrad.nn import MLP

xs = [
    [2.0, 3.0],
    [-1.0, 1.0],
    [0.5, -2.0],
    [1.0, 1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(2, [12, 12, 1])

step = 0.01

for epoch in range(10000):
    # forward pass
    ypred = [model(x)[0] for x in xs]

    loss = sum(((pred - actual) ** 2 for pred, actual in zip(ypred, ys))) / len(ys)

    model.zero_grad()

    loss.backward()

    for param in model.parameters():
        param.data -= step * param.grad

    print(f"Epoch: {epoch}, Loss: {loss.data}")
