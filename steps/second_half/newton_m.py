import numpy as np
from deep3framework import Variable

def f(x):
    y = x**4 - 2 * x**2
    return y

# deep3framework에서 2차 미분은 자동으로 구하지 못하므로 수동으로 입력
def gx2(x):
    return 12 * x**2 - 4

if __name__ == '__main__':
    '''
    # 수동
    print("Newton - Manual Approach")
    x1 = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print("{} 번째 x{}".format(i, x1))

        y = f(x1)
        x1.cleargrad()
        y.backward()

        x1.data -= x1.grad / gx2(x1.data)
    '''
    x2 = Variable(np.array(2.0))
    y = f(x2)
    y.backward(create_graph=True)
    print(x2.grad)

    # 두번째 역전파 진행(2차 미분)
    gx = x2.grad
    x2.cleargrad()
    gx.backward()
    print(x2.grad)

    # 자동
    print("Newton - Auto")
    x3 = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print("{} 번째 x{}".format(i, x3))

        y = f(x3)
        x3.cleargrad()
        y.backward(create_graph=True)

        gx = x3.grad
        x3.cleargrad()
        gx.backward()
        gx2 = x3.grad

        x3.data -= gx.data / gx2.data