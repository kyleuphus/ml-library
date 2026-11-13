# mygrad — NumPy Autodiff and Neural Network Library

mygrad implements reverse-mode automatic differentiation and a small neural network library in Python. I started with a scalar engine, then built a NumPy-backed tensor engine to compute gradients for entire batches. The tensor engine trains a 235,146-parameter MNIST classifier to **97.5% accuracy on 14,000 held-out images** using the recipe below.

- **Autodiff.** Each operation records its inputs and a local backward rule. A reverse-topological traversal accumulates contributions from every path through the computation graph. [Tensor engine](src/mygrad/engine.py)
- **Batched gradients.** A dense layer applies `X @ W + b` to a batch at once. Backpropagation sums the gradient over the examples that share the same bias vector. [Dense layers](src/mygrad/nn/mlp.py)
- **Training.** The library provides dense layers with Xavier initialization, a configurable MLP, activation and loss functions, and mini-batch SGD with shuffling and optional learning-rate decay. [Training loop](src/mygrad/training/loop.py) · [Optimizer](src/mygrad/optim/sgd.py)

## Quick start

From the repository root on macOS or Linux, create an environment and install NumPy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Run this example from the same directory. It differentiates a sum of squares through a bias shared by two rows.

```bash
python - <<'PY'
import numpy as np
from mygrad.engine import Tensor

x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
bias = Tensor(np.array([0.5, -0.5]))
y = x + bias
loss = (y * y).sum()
loss.grad = np.ones_like(loss.data)
loss.backward()

np.testing.assert_allclose(bias.grad, [10.0, 10.0])
print(bias.grad)  # [10. 10.]
PY
```

The caller seeds the output gradient before calling `backward()`. Use floating-point tensors for inputs that participate in differentiation.

## Reproduce the MNIST result

The run below correctly classified **13,649 of 14,000** held-out images with Python 3.13.7, NumPy 2.4.4, and scikit-learn 1.9.0. It uses a shuffled 80/20 split of OpenML's 70,000-image MNIST dataset. Scikit-learn loads and splits the dataset. mygrad implements the network, loss, backpropagation, and parameter updates.

From the same environment and repository root, install scikit-learn and run the recipe. The first run downloads the dataset from OpenML.

```bash
python -m pip install scikit-learn
python - <<'PY'
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mygrad.engine import Tensor
from mygrad.nn.activations import relu
from mygrad.nn.losses import softmax_cross_entropy
from mygrad.nn.mlp import MLP
from mygrad.training.loop import train

np.random.seed(42)
pixels, labels = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    pixels.astype(np.float64) / 255.0, labels.astype(int),
    test_size=0.2, random_state=42,
)

model = MLP(784, [256, 128], 10, activation=relu)
train(model, x_train, np.eye(10)[y_train], softmax_cross_entropy,
      lr=0.01, epochs=80, batch_size=64)

predicted = model(Tensor(x_test)).data.argmax(axis=1)
print(f"Held-out accuracy: {(predicted == y_test).mean():.2%}")
PY
```

## How backpropagation works

1. **Record the forward pass.** Each `Tensor` stores its data, gradient buffer, parent tensors, and a backward function. For example, `Y = X @ W + b` creates separate nodes for matrix multiplication and addition.
2. **Walk the graph backward.** Depth-first search visits each node once. Running the backward functions in reverse-topological order ensures a shared node receives all contributions before propagating them to its inputs.
3. **Reduce broadcast gradients.** Each example uses the same bias. The addition rule reduces the batch-shaped gradient back to the bias vector's shape by summing across the batch.
4. **Update parameters.** The training loop clears the parameter gradients, computes the loss, seeds the output gradient, and calls backpropagation. SGD then updates the weights in place.

## Verification and scope

- The quick start checks a concrete bias gradient, and the MNIST recipe checks the full training path with held-out data.
- The [legacy tests](tests/) compare parts of the original scalar engine with finite differences and PyTorch. Their imports still reference the scalar engine's old location, and they currently fail during collection. The [scalar implementation](archive/scalar/) is preserved for reference.
- The tensor implementation targets CPU-based dense networks. The bundled training path uses the fused softmax-cross-entropy backward rule. Standalone softmax backpropagation and some mixed-rank broadcasting cases need further work. The fused loss currently prints a batch sum and uses a batch-averaged gradient, so its logged value depends on the batch size.
- [The MNIST script](examples/mnist.py) and [Iris notebook](examples/iris.ipynb) record earlier experiments. Use the recipe above to reproduce the reported MNIST result.

## References

The scalar engine was inspired by [micrograd](https://github.com/karpathy/micrograd). I also used [PyTorch's autograd notes](https://pytorch.org/docs/stable/notes/autograd.html) and [tinygrad](https://github.com/tinygrad/tinygrad) as learning references.

