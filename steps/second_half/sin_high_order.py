if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from deep3framework import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)

    for i in range(3):
        gx1 = x.grad
        x.cleargrad()
        gx1.backward(create_graph=True)
        print(x.grad) # n차 미분

    print("Plot Drawing")
    # -7부터 7까지 균일하게 200등분한 배열 생성
    x2 = Variable(np.linspace(-7,7,200))
    y2 = F.sin(x2)
    y2.backward(create_graph=True)

    logs = [y2.data]

    for i in range(3):
        gx2 = x2.grad
        x2.cleargrad()
        gx2.backward(create_graph=True)
        logs.append(x2.grad.data)

    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        # def plot(*args, scalex=True, scaley=True, data=None, **kwargs)
        plt.plot(x2.data, logs[i], label=labels[i])
    plt.legend(loc='lower right')
    plt.show()