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
    def __init__(self, axes=None):
        self.axes = axes
    def forward(self, x):
        y = x.transpose(self.axes)
        return y
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        # gy의 형상을 미세하게 조정
        # axis, keepdims를 활용함으로써 기울기의 형상을 변환하는 경우가 생기기 때문에 그에 대응
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
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

def transpose(x, axes=None):
    return Transpose(axes)(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
