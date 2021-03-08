import numpy as np
from deep3framework.core import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = Variable(np.array([[1,2,3],[4,5,6]]))
    y = F.transpose(x)
    y.backward()
    print(x.grad.shape)
    print(y.shape)
    print("*\n")
    x1 = Variable(np.random.rand(2,3))
    # y1 = x1.transpose()
    y1 = x1.T
    y1.backward()
    print(x1.grad.shape)
    print(y1.shape)
    print("*\n")
    print("Apply to Variable")
    x2 = Variable(np.random.rand(1,2))
    y2 = x2.transpose()
    y2.backward()
    print("---")
    print(x2.grad.shape)
    print("---")
    print(y2.shape)