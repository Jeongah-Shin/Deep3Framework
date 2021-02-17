import numpy as np
from deep3framework import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    # 아래 과정에서 y의 기울기도 자동으로 채워진다.
    # 단, y.grad.shape == y.data.shape
    y.backward(retain_grad=True)
    print(y)
    print(x.grad)

