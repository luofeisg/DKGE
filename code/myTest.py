import numpy as np
from scipy import sparse
import collections
import torch
import torch.nn as nn

a = torch.tensor([[10, 10, 10], [1, 1, 1], [0.1, 0.1, 0.1], [0.01, 0.01, 0.01]])
index = torch.tensor([[2, 1, 2, 0]])
result = torch.zeros(5, 5)
result.scatter_add_(1, index, a)

print()