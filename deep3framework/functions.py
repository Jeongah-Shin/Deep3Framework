import numpy as np
from deep3framework.core import Function

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx
def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self,x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
def cos(x):
    return Cos()(x)