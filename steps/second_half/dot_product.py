import numpy as np

# dot product of vector
# 인수가 모두 1차원 배열일 때
a1 = np.array([1, 2, 3])
b1 = np.array([4, 5, 6])
c1 = np.dot(a1, b1)
print(c1)

# multiplication of matrix
# 인수가 2차원 배열일 때
a2 = np.array([[1, 2], [3, 4]])
b2 = np.array([[5, 6], [7, 8]])
c2 = np.dot(a2, b2)
print(c2)
