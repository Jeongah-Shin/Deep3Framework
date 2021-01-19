import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

if __name__ == '__main__':
    data = np.array(1.0)
    x = Variable(data)

    x.data = np.array(2.0)
    print(x.data)

    # Scalar - 0 dimensional Tensor
    a = np.array(1)
    print(a.ndim)

    # Vector - 1 dimensional Tensor
    b = np.array([1,2,3])
    print(b.ndim)

    # Matrix - 2 dimensional Tensor
    c = np.array([[1,2,3],
                  [4,5,6]])
    print(c.ndim)