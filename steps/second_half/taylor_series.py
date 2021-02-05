import numpy as np
import math
from deep3framework import Function
from deep3framework import Variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1)**i / math.factorial(2*i + 1)
        t = c * x**(2*i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

if __name__ == '__main__':
    print("Sin Function")
    x = Variable(np.array(np.pi/4))
    # y = sin(x)
    y = my_sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
