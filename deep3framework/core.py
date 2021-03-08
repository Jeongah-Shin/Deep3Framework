import contextlib
import weakref
import numpy as np
import deep3framework

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_val = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_val)

def no_grad():
    return using_config('enable_backprop', False)

class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
               raise TypeError('{}의 타입은 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    def __str__(self):
        # data = np.array_str(self.data).replace('\n', '')
        data = self.data
        grad = None
        if self.grad is not None:
            grad = self.grad

        creator = self.creator
        return '\n\tVaraible - data: {}, grad: {}, creator: {}'.format(data, grad, creator)
    def cleargrad(self):
        self.grad = None
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    # *shape - 가변 길이 인수 지원
    def reshape(self, *shape):
        # 지정된 *shape Tuple 혹은 List 일때
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return deep3framework.functions.reshape(self, shape)
    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple,list)) or axes[0] is None:
                axes = axes[0]
        return deep3framework.functions.transpose(self, axes)
    def sum(self, axis=None, keepdims=False):
        return deep3framework.functions.sum(self, axis, keepdims)
    # 인스턴스 변수로 활용
    # y = x.T
    @property
    def T(self):
        return deep3framework.functions.transpose(self)
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Variable(np.ones_like(self.data))
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            f = funcs.pop()

            gys = [output().grad for output in f.outputs]
            # 역전파를 1회만 한다면(처음 이후로 다시 할 일이 없다면) 역전파 계산도 '역전파 비활성 모드'로 실행
            # 2차 이싱의 미분이 필요하다면 create_graph = True 로 설정
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys) # 메인 backward

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx # 이 계산도 해당
                    if x.creator is not None:
                        add_func(x.creator)
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape: # for broadcast
            gx0 = deep3framework.functions.sum_to(gx0, self.x0_shape)
            gx1 = deep3framework.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape: # for broadcast
            gx0 = deep3framework.functions.sum_to(gx0, x0.shape)
            gx1 = deep3framework.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape: # for broadcast
            gx0 = deep3framework.functions.sum_to(gx0, self.x0_shape)
            gx1 = deep3framework.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape: # for broadcast
            gx0 = deep3framework.functions.sum_to(gx0, x0.shape)
            gx1 = deep3framework.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        # x = self.inputs[0].data
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)
def rdiv(x0, x1):
    x1 = as_array(x1)
def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow