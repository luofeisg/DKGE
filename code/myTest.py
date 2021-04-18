import numpy as np
from scipy import sparse
import collections
import torch
import torch.nn as nn


embedding = np.random.rand(10,2)
samples = [7, 2, 8, 5, 3, 7, 8]

# o = np.array([[1,11],
#              [2,22],
#              [3,33],
#              [4,44],
#              [5,55],
#              [6,66],
#              [7,77],
#              [8,88],
#              [9,99]], dtype=float)
o = np.random.rand(10,2)
vectors = o[samples]
embedding[samples] = vectors

entity_unique, return_index, return_inverse = np.unique(samples, return_index=True, return_inverse=True)
print('samples:', samples)
print('entity_unique:', entity_unique)
print('return_index:', return_index)
print('return_inverse:', return_inverse)

print()