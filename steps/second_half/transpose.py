import numpy as np
from deep3framework.core import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = Variable(np.array([[1,2,3],[4,5,6]]))
    y = F.transpose(x)
    y.backward()
    print(x.grad)

    x1 = Variable(np.random.rand(2,3))
    y1 = x1.transpose()
    y1 = x1.T
    print(x1)
    print(y1)