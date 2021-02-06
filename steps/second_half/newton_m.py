import numpy as np
from deep3framework import Variable

def f(x):
    y = x**4 - 2 * x**2
    return y

# deep3framework에서 2차 미분은 자동으로 구하지 못하므로 수동으로 입력
def gx2(x):
    return 12 * x**2 - 4

if __name__ == '__main__':
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print("{} 번째 x{}".format(i, x))

        y = f(x)
        x.cleargrad()
        y.backward()

        x.data -= x.grad / gx2(x.data)