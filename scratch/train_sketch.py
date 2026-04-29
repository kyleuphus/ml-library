"""
1) determine hidden layer count and number of neurons in each layer
2) forward pass inputs
3) backprop from output layer
4) determine loss
5) nudge each weight and bias by small amount to decrease loss
6) repeat until small loss
"""

from mygrad.engine import Value
from mygrad.nn import MLP

# 2 input -> 3 -> 3 -> 1 output
model = MLP(2, [3, 3, 1])

# data
xs = [[1.5, -2.0], [-3.2, 1.8], [4.1, 0.0]]
ys = [1.0, -1.0, 1.0]

step = 0.01

for epoch in range(20):
    # forward pass:
    ypred = [model(x)[0] for x in xs]

    loss = sum(((pred - actual) ** 2 for pred, actual in zip(ypred, ys))) / len(ys)

    if epoch == 0:
        print(f"loss = {loss}")
        print(f"type(loss) = {type(loss)}")

    # zero gradients:
    for layer in model.layers:
        for neuron in layer.neurons:
           for w in neuron.w:
                w.grad = 0
            neuron.b.grad = 0

    # backward pass:
    loss.backward()

    # update weights:
    for layer in model.layers:
        for neuron in layer.neurons:
            for w in neuron.w:
                w.data -= step * w.grad
            neuron.b.data -= step * neuron.b.grad

    print(f"Epoch: {epoch}, Loss: {loss.data}")
