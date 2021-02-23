import numpy as np
from deep3framework.core import Function
from deep3framework.core import as_variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

class Cos(Function):
    def forward(self,x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y*y)
        return gx

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        # 입력 x의 형상을 기억
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    def backward(self, gy):
        return reshape(gy, self.x_shape)

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    def backward(self, gy):
        gx = transpose(gy)
        return gx

def cos(x):
    return Cos()(x)

def sin(x):
    return Sin()(x)

def tanh(x):
    return Tanh()(x)

def reshape(x, shape):
    # 인수 x는 ndarray 혹은 Variable 인스턴스라고 가정
    # 형상이 이미 같다면 아무일도 하지 않고 x를 그대로 return
    if x.shape == shape:
        # 언제나 Variable 인스턴스를 반환하는 것을 보장
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x):
    return Transpose()(x)

