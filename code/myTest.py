import numpy as np
from scipy import sparse
import collections

matrix = np.eye(6)

sparse_matrix = sparse.csr_matrix(matrix)

dd = collections.defaultdict()

print("对角矩阵: \n{}".format(matrix))

print("\nsparse存储的矩阵: \n{}".format(sparse_matrix))