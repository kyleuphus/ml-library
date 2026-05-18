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
import torch.nn as nn
import torch.nn.init as init

# %%
input = np.array([2, 1])

weights_one = np.array([[.7, -1.1], 
                        [1.4, 2.3]])

biases_one = np.array([0.3, .55])

hidden_out = np.array([0, 0])

weights_two = np.array([[1.1, -1.3]])

biases_two = np.array([.3])

output = np.array([0])

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

hidden_out = np.maximum(0, weights_one.dot(input) + biases_one)
p_hidden_out = torch.relu(torch.matmul(p_weights_one, p_input).add(p_biases_one))

output = weights_two.dot(hidden_out) + biases_two
p_output = torch.matmul(p_weights_two,p_hidden_out).add(p_biases_two)
p_output.retain_grad()
    

# %%
print(hidden_out)
print(output)

# %%
target = np.array([2.0])
p_target = torch.tensor(target)

mse = (output - target) ** 2
p_mse = (p_output.subtract(p_target)).pow(2)

print(mse)

# %%
# gradient of loss function

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
