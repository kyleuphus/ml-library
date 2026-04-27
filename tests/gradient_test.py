import math
import torch
from torch import nn
from mygrad.nn import MLP
from mygrad.engine import Value


def test_1():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + a
    c.backward()

    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(-3.0, requires_grad=True)
    c_t = a_t * b_t + a_t
    c_t.backward()

    assert math.isclose(a.grad, a_t.grad.item(), rel_tol=1e-5)
    assert math.isclose(b.grad, b_t.grad.item(), rel_tol=1e-5)


def test_2():
    a = Value(2.0)
    b = Value(-3.0)
    c = ((a / b) ** 2).exp()
    c.backward()

    a_t = torch.tensor(2.0, requires_grad=True)
    b_t = torch.tensor(-3.0, requires_grad=True)
    c_t = ((a_t / b_t) ** 2).exp()
    c_t.backward()

    assert math.isclose(a.grad, a_t.grad.item(), rel_tol=1e-5)
    assert math.isclose(b.grad, b_t.grad.item(), rel_tol=1e-5)


def test_relu():
    a = Value(5.2)
    b = Value(0.5)
    c = (-a + b).relu()
    c.backward()

    a_t = torch.tensor(5.2, requires_grad=True)
    b_t = torch.tensor(0.5, requires_grad=True)
    c_t = (-a_t + b_t).relu()
    c_t.backward()

    assert math.isclose(a.grad, a_t.grad.item(), rel_tol=1e-5)
    assert math.isclose(b.grad, b_t.grad.item(), rel_tol=1e-5)


def test_tanh():
    a = Value(5.2)
    b = Value(0.5)
    c = (a * b).tanh()
    c.backward()

    a_t = torch.tensor(5.2, requires_grad=True)
    b_t = torch.tensor(0.5, requires_grad=True)
    c_t = (a_t * b_t).tanh()
    c_t.backward()

    assert math.isclose(a.grad, a_t.grad.item(), rel_tol=1e-5)
    assert math.isclose(b.grad, b_t.grad.item(), rel_tol=1e-5)


def test_MLP_forward():
    xs = [[2.0, 4.0], [1.5, 3.0]]

    model = MLP(2, [1])

    model_p = nn.Linear(2, 1)

    weights = torch.tensor([weight.data for weight in model.layers[0].neurons[0].w])
    biases = torch.tensor([model.layers[0].neurons[0].b.data])
    with torch.no_grad():
        model_p.weight.copy_(weights)
        model_p.bias.copy_(biases)

    y_pred = model(xs[0])[0]
    y_pred_p = model_p(torch.tensor(xs[0]))

    y_pred.backward()
    y_pred_p.backward()

    assert math.isclose(y_pred.data, y_pred_p.item(), rel_tol=1e-5)
    assert math.isclose(
        model.layers[0].neurons[0].w[0].grad,
        model_p.weight.grad[0][0].item(),
        rel_tol=1e-5,
    )
