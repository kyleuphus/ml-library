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
