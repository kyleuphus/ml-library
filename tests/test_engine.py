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
