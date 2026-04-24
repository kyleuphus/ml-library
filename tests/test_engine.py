import math
from mygrad.engine import Value


def test_add_basic():
    a = Value(-2.0)
    b = Value(4.0)
    c = a + b
    assert c.data == 2.0


def test_add_large():
    a = Value(127495.0)
    b = Value(68252.0)
    c = a + b
    assert c.data == 195747.0


def test_add_doubleneg():
    a = Value(-14.0)
    b = Value(-6.0)
    c = a + b
    assert c.data == -20.00


def test_radd_basic():
    a = Value(5.0)
    b = 2.0 + a
    assert b.data == 7.0


def test_radd_negative():
    a = Value(-5.0)
    b = -10.0 + a
    assert b.data == -15.0


def test_sub_basic():
    a = Value(10.0)
    b = Value(5.0)
    c = a - b
    assert c.data == 5.0


def test_sub_toneg():
    a = Value(5.0)
    b = Value(10.0)
    c = a - b
    assert c.data == -5.0


def test_sub_neg():
    a = Value(5.0)
    b = Value(-10.0)
    c = a - b
    assert c.data == 15.0


def test_rsub_basic():
    a = Value(10.0)
    b = 20.0 - a
    assert b.data == 10.0


def test_rsub_neg():
    a = Value(-5.0)
    b = 10 - a
    assert b.data == 15


def test_neg():
    a = Value(10.0)
    b = -a
    assert b.data == -10.0


def test_neg_topos():
    a = Value(-10.0)
    b = -a
    assert b.data == 10.0


def test_mul_basic():
    a = Value(5.0)
    b = Value(2.0)
    c = a * b
    assert c.data == 10.0


def test_mul_negative():
    a = Value(-10.0)
    b = Value(10.0)
    c = a * b
    assert c.data == -100.0


def test_mul_doubleneg():
    a = Value(-4.0)
    b = Value(-3.0)
    c = a * b
    assert c.data == 12.0


def test_rmul_basic():
    a = Value(5.0)
    b = 2.0 * a
    assert b.data == 10.0


def test_rmul_neg():
    a = Value(5.0)
    b = -2.0 * a
    assert b.data == -10.0


def test_truediv_basic():
    a = Value(100.0)
    b = Value(10.0)
    c = a / b
    assert c.data == 10.0


def test_truediv_neg():
    a = Value(10.0)
    b = Value(-2.0)
    c = a / b
    assert c.data == -5.0


def test_rtruediv_basic():
    a = Value(10.0)
    b = 50.0 / a
    assert b.data == 5.0


def test_rtruediv_neg():
    a = Value(15.0)
    b = -30.0 / a
    assert b.data == -2.0


def test_exp_basic():
    a = Value(2.0)
    b = a.exp()
    assert b.data == math.exp(2)


def test_pow_basic():
    a = Value(2.0)
    b = a**4
    assert b.data == 16.0


def test_pow_neg():
    a = Value(-2.0)
    b = a**3
    assert b.data == -8.0


def test_tanh_basic():
    a = Value(2.0)
    b = a.tanh()
    expected = (math.exp(2) - math.exp(-2)) / (math.exp(2) + math.exp(-2))
    assert math.isclose(b.data, expected)


def test_tanh_zero():
    a = Value(0.0)
    b = a.tanh()
    assert b.data == 0.0


def test_relu_neg():
    a = Value(-1.0)
    b = a.relu()
    assert b.data == 0.0


def test_relu_pos():
    a = Value(10.0)
    b = a.relu()
    assert b.data == 10.0


def test_relu_zero():
    a = Value(0.0)
    b = a.relu()
    assert b.data == 0.0


def test_gradients_simple():
    h = 1e-4

    # f(x + h)
    a_plus = Value(3.0 + h)
    output_plus = ((a_plus * 3.0) / -2.0).tanh()

    # f(x)
    a_original = Value(3.0)
    output_original = ((a_original * 3.0) / -2.0).tanh()

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_plus.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert (numerical_grad - analytical_grad) < 1e-6


def test_gradients_add():
    h = 1e-4

    # f(x + h)
    a_plus = Value(-1.0 + h)
    output_plus = a_plus + 10.0

    # f(x)
    a_original = Value(-1.0)
    output_original = a_original + 10

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_plus.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_grdients_mult():
    h = 1e-4

    # f(x + h)
    a_mult = Value(2.0 + h)
    output_mult = a_mult * -5.0

    # f(x)
    a_original = Value(2.0)
    output_original = a_original * -5.0

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_mult.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_subtract():
    h = 1e-4

    # f(x + h)
    a_subtract = Value(5.0 + h)
    output_subtract = a_subtract - 12.0

    # f(x)
    a_original = Value(5.0)
    output_original = a_original - 12.0

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_subtract.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_div():
    h = 1e-4

    # f(x + h)
    a_div = Value(26.0 + h)
    output_div = a_div / 2.0

    # f(x)
    a_original = Value(26.0)
    output_original = a_original / 2.0

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_div.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_pow():
    h = 1e-4

    # f(x + h)
    a_pow = Value(8.0 + h)
    output_pow = a_pow ** (3 / 8)

    # f(x)
    a_original = Value(8.0)
    output_original = a_original ** (3 / 8)

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_pow.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_exp():
    h = 1e-7

    # f(x + h)
    a_exp = Value(2.0 + h)
    output_exp = a_exp.exp()

    # f(x)
    a_original = Value(2.0)
    output_original = a_original.exp()

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_exp.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_tanh():
    h = 1e-4

    # f(x + h)
    a_tanh = Value(8.0 + h)
    output_tanh = a_tanh.tanh()

    # f(x)
    a_original = Value(8.0)
    output_original = a_original.tanh()

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_tanh.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_relu():
    h = 1e-4

    # f(x + h)
    a_relu = Value(-8.0 + h)
    output_relu = a_relu.relu()

    # f(x)
    a_original = Value(-8.0)
    output_original = a_original.relu()

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_relu.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6


def test_gradients_relu_pos():
    h = 1e-4

    # f(x + h)
    a_relu = Value(8.0 + h)
    output_relu = a_relu.relu()

    # f(x)
    a_original = Value(8.0)
    output_original = a_original.relu()

    # (f(x + h) - f(x)) / h
    numerical_grad = (output_relu.data - output_original.data) / h

    # backprop algorithm
    output_original.backward()
    analytical_grad = a_original.grad

    assert abs((numerical_grad - analytical_grad)) < 1e-6
