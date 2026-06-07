# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import torch
from src.mygrad.engine import Tensor

# %%
input = np.array([2, 1])
t_input = Tensor([2.0, 1.0])

weights_one = np.array([[.7, -1.1], 
                        [1.4, 2.3]])
t_weights_one = Tensor([[.7, -1.1], [1.4, 2.3]])

biases_one = np.array([0.3, .55])
t_biases_one = Tensor([.3, .55])

hidden_out = np.array([0, 0])
t_hidden_out = Tensor([0.0, 0.0])

weights_two = np.array([[1.1, -1.3]])
t_weights_two = Tensor([1.1, -1.3])

biases_two = np.array([.3])
t_biases_two = Tensor([.3])

output = np.array([0])
t_output = Tensor([0.0])


p_input = torch.tensor(input, dtype=torch.float64, requires_grad=True)
p_weights_one = torch.tensor(weights_one, dtype=torch.float64, requires_grad=True)
p_biases_one = torch.tensor(biases_one, dtype=torch.float64, requires_grad=True)
p_hidden_out = torch.tensor(hidden_out, dtype=torch.float64, requires_grad=True)
p_weights_two = torch.tensor(weights_two, dtype=torch.float64, requires_grad=True)
p_biases_two = torch.tensor(biases_two, dtype=torch.float64, requires_grad=True)
p_output = torch.tensor(output, dtype=torch.float64, requires_grad=True)

# %%
print(weights_one.shape)
print(input.shape)
print(input.dot(weights_one))

pre_activation = weights_one.dot(input) + biases_one
t_pre_activation = t_weights_one @ t_input + t_biases_one

hidden_out = np.maximum(0, weights_one.dot(input) + biases_one)
t_hidden_out = t_pre_activation.relu()

p_hidden_out = torch.relu(torch.matmul(p_weights_one, p_input).add(p_biases_one))

output = weights_two.dot(hidden_out) + biases_two
t_output = t_weights_two @ t_hidden_out + t_biases_two

p_output = torch.matmul(p_weights_two,p_hidden_out).add(p_biases_two)
p_output.retain_grad()


# %%
print(hidden_out)
print(output)
print(t_output.data)

# %%
target = np.array([2.0])
p_target = torch.tensor(target)
t_target = Tensor([2.0])

mse = (output - target) ** 2
p_mse = (p_output.subtract(p_target)).pow(2)
t_mse = (t_output - t_target) * (t_output - t_target)
print(t_mse.data.shape)
print(t_mse._op)
t_mse.grad = np.array([1.0])
print(mse)
print(t_mse.data)

# %%
# manually computed gradient of loss function

d_weights_one = np.array([[-36.894, -18.447], [43.602, 21.801]])
d_biases_one = np.array([-18.447, 21.801])
d_weights_two = np.array([-10.062, -94.7505])
d_biases_two = np.array([-16.77])
d_output = np.array([-16.77])

# %%
# checking against pytorch
p_mse.backward()
assert(np.allclose(d_weights_one, p_weights_one.grad.numpy()))
assert(np.allclose(d_biases_one, p_biases_one.grad.numpy()))
assert(np.allclose(d_weights_two, p_weights_two.grad.numpy()))
assert(np.allclose(d_biases_two, p_biases_two.grad.numpy()))

# %%
t_mse.backward()
print(d_weights_one)
print(t_weights_one.grad)
assert(np.allclose(d_weights_one, t_weights_one.grad))
assert(np.allclose(d_biases_one, t_biases_one.grad))
assert(np.allclose(d_weights_two, t_weights_two.grad))
assert(np.allclose(d_biases_two, t_biases_two.grad))

# %%
# computing gradients with numpy
d_loss_d_output = 2 * (output - target)
d_loss_d_biases_two = d_loss_d_output * 1
d_loss_d_weights_two = d_loss_d_output * hidden_out
d_loss_d_biases_one = d_loss_d_output * weights_two * (pre_activation > 0).astype(float)
d_loss_d_weights_one = d_loss_d_output * np.outer(weights_two * (pre_activation > 0).astype(float), input)

assert(np.allclose(d_loss_d_weights_one, p_weights_one.grad.numpy()))
assert(np.allclose(d_loss_d_biases_one, p_biases_one.grad.numpy()))
assert(np.allclose(d_loss_d_weights_two, p_weights_two.grad.numpy()))
assert(np.allclose(d_loss_d_biases_two, p_biases_two.grad.numpy()))
