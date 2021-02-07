import numpy as np
from deep3framework import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(5):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad) # n차 미분