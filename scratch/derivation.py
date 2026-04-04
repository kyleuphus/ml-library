def f(x):
    return 3 * x**2 - 4 * x + 5


x = 3.0
h = 0.0001

slope = ((f(x + h)) - f(x)) / h

print(slope)
