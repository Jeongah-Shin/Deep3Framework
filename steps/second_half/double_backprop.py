import numpy as np
from deep3framework import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    # x = Variable(np.array(2.0))
    # y = x ** 2
    # y.backward(create_graph=True)
    # gx = x.grad
    # x.cleargrad()
    #
    # z = gx**3 + y
    # z.backward()
    # print(x.grad)

    print("Hessian-vector product")
    x = Variable(np.array([1.0, 2.0]))
    v = Variable(np.array([4.0, 5.0]))

    def f(x):
        t = x**2
        y = F.sum(t)
        return y

    y = f(x)
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()

    z = F.matmul(v,gx)
    z.backward()
    print(x.grad)