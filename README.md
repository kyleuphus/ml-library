# mygrad

A small deep-learning library built from scratch in Python — a reverse-mode automatic differentiation engine plus the neural-network layers, losses, and optimizer needed to train real models on top of it. The goal was to understand how frameworks like PyTorch actually work by rebuilding the core myself, not just calling `loss.backward()`.

Trained on MNIST, a multi-layer perceptron built entirely on this engine reaches **~97.8% test accuracy**.

## What it demonstrates

- **Reverse-mode autodiff from first principles** — a `Tensor` type that records a computation graph during the forward pass and walks it in reverse-topological order to accumulate gradients.
- **Correct gradient math, including broadcasting.** Each operation defines its own local backward rule; an `unbroadcast` step reduces gradients back to the original operand shapes so element-wise ops and bias terms differentiate correctly.
- **A PyTorch-style API** built on top: `Module`, `Linear`, `MLP`, activation functions, loss functions, and an SGD optimizer with a mini-batch training loop.
- **Verified against PyTorch.** Gradients were checked against PyTorch's autograd as the ground-truth baseline.

The project evolved in two stages, both preserved in the repo: it began as a scalar-valued engine in the spirit of [micrograd](https://github.com/karpathy/micrograd) (now in `archive/scalar/`), then graduated to a vectorized, NumPy-backed tensor engine that supports batched training (`src/mygrad/`).

## Features

**Autograd engine** (`src/mygrad/engine.py`)
- `Tensor` wrapping a NumPy array, with `.data`, `.grad`, and a recorded graph
- Operations: matmul, add, subtract, multiply, divide, negate, sum, log — each with its own backward rule
- Broadcasting-aware gradient accumulation
- Topological-sort backward pass

**Neural network** (`src/mygrad/nn/`)
- `Module` base class with `parameters()` and `zero_grad()`
- `Linear` layer with Xavier (Glorot) weight initialization
- `MLP` — configurable depth/width feed-forward network
- Activations: ReLU, Leaky ReLU, tanh, sigmoid, softmax
- Losses: MSE, binary cross-entropy, categorical cross-entropy, and a fused softmax–cross-entropy (with the simplified `softmax(x) - target` gradient)

**Optimization & training** (`src/mygrad/optim/`, `src/mygrad/training/`)
- SGD optimizer
- Mini-batch training loop with per-epoch shuffling and learning-rate decay

## Quickstart

```python
import numpy as np
from mygrad.engine import Tensor
from mygrad.nn.mlp import MLP
from mygrad.nn.activations import relu
from mygrad.nn.losses import softmax_cross_entropy
from mygrad.training.loop import train

# 784 -> 256 -> 128 -> 10 classifier
mlp = MLP(784, [256, 128], 10, activation=relu)

train(mlp, X_train, y_train, softmax_cross_entropy,
      lr=0.01, epochs=30, batch_size=64)

preds = np.argmax(mlp(Tensor(X_test)).data, axis=1)
```

Worked end-to-end examples are in [`examples/`](examples/):
- `examples/mnist.py` — handwritten-digit classification (~97.8% test accuracy)
- `examples/iris.py` — the Iris dataset

## Project structure

```
src/mygrad/
  engine.py            # Tensor + autograd
  nn/
    mlp.py             # Module, Linear, MLP
    activations.py     # relu, tanh, sigmoid, leaky_relu, softmax
    losses.py          # mse, cross-entropy variants
  optim/sgd.py         # SGD
  training/loop.py     # mini-batch training loop
archive/scalar/        # earlier scalar-valued (micrograd-style) engine
examples/              # MNIST and Iris training scripts
tests/                 # gradient checks against PyTorch
```

## Scope & limitations

This is an educational library. It's NumPy-backed and prioritizes clarity over speed — there's no GPU support, no convolution layers, and a single optimizer.

## References

Built while studying [micrograd](https://github.com/karpathy/micrograd), the [PyTorch autograd internals](https://pytorch.org/docs/stable/notes/autograd.html), and [tinygrad](https://github.com/tinygrad/tinygrad).
