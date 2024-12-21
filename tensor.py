import torch
import numpy as np

# x = torch.rand(5, 3)
# print(type(x))

# x = torch.ones(5, 3 , 2 , dtype=torch.bool)
# print(x.shape)
# print(x.dtype)
# print(x)

# torch.manual_seed(1729)
# r1 = torch.rand(2, 2)
# print('A random tensor:')
# print(r1)

# torch.manual_seed(1729)
# r2 = torch.rand(3, 3)
# print('A different random tensor:')
# print(r2)

# print('Are they the same?')
# print(r1 , r2)

x = torch.empty(3, 4)
print(type(x))
print(x)

some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)


# Create a numpy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
print("Numpy array:")
print(numpy_array)

# Convert numpy array to tensor
tensor_from_numpy = torch.from_numpy(numpy_array)
print("\nTensor from numpy array:")
print(tensor_from_numpy)


