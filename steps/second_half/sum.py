import numpy as np
from deep3framework import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = np.array([[1,2,3],[4,5,6]])
    x2 = np.random.randn(1,2,3)
    y = np.sum(x, axis=(0), keepdims=True)
    print(y)
    y2 = np.sum(x2, axis=(0,1))
    print(x2)
    print("----")
    print(np.sum(x2, axis=(0)))
    print("----")
    print(np.sum(x2, axis=(1)))
    print("----")
    print(np.sum(x2, axis=(2)))
    print("----")
    print(y2)
    print("\n\n\n")
    xv = Variable(np.array([[1,2,3],[4,5,6]]))
    yv = F.sum(xv, axis=0)
    yv.backward()
    print(yv)
    print(xv.grad)

    xv_1 = Variable(np.random.randn(2, 3, 4, 5))
    yv_1 = xv_1.sum(keepdims=True)
    print(yv_1.shape)
