import numpy as np
import weakref
from steps import config as c, variable as v


# Function 클래스는 기반 클래스로서 모든 함수에 공통되는 기능을 구현
# 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현

class Function:
    # input 은 Variable 인스턴스라고 가정
    # * notation을 통해 가변 길이 인수를 건네 함수를 호출
    def __call__(self, *inputs):
        # Variable이 아닌 input이 들어왔을 때, Variable로 매핑
        # ndarray, int, float 등이 input으로 들어오는 경우를 대비
        inputs = [as_variable(x) for x in inputs]
        # 데이터 꺼내기, List Comprehension(리스트 내포) 사용
        xs = [x.data for x in inputs]
        # *xs ---> List Unpack(리스트 언팩)
        # 리스트 원소를 낱개로 풀어서 전달하는 기법
        # xs = [x0, x1]일 때 self.forward(*xs)는 self.forward(x0, x1)과 같다.
        ys = self.forward(*xs)
        # ys가 튜플이 아닌 경우 튜플로 변경
        if not isinstance(ys, tuple):
            ys = (ys, )
        # 0차원의 x = ndarray - np.array(1.0)
        # x ** 2를 하면 np.float64 or np.float32로 리턴하는 문제 해결
        outputs = [v.Variable(as_array(y)) for y in ys]
        if c.Config.enable_backprop:
            # 입련 변수의 generation을 그대로 수용하되,
            # 입력 변수가 여러개라면, 가장 큰 generation 수를 채택
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                # 출력 변수에 창조자를 설정
                output.set_creator(self)
            # 입력 변수를 보관해놨다가 역전파시 사용
            self.inputs = inputs
            # 출력도 저장
            self.outputs = [weakref.ref(output) for output in outputs]

        # outputs에 원소가 하나뿐이면 리스트가 아니라 그 원소(해당 변수)만 반환
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        # 해당 메서드는 상속하여 구해야 한다.
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError()

# 입력이 스칼라인 경우 ndarray 인스턴스로 변환
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

class Square(Function):
    # Function 클래스를 상속하기 때문에
    # __call__ 메서드는 그대로 계승
    def forward(self, x):
        y = x ** 2
        return y
    # gy 는 ndarray 인스턴스
    def backward(self, gy):
        # x = self.input.data
        # 가변 길이 인수 지원하도록 수정
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    def backward(self, gy):
        # 출력쪽에서 전해지는 미분값에 1을 곱한 값 == 그대로 흘려보내는 것
        return gy, gy

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    def backward(self, gy):
        return gy, -gy

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function):
    # c는 상수로 취급하여 따로 미분 계산 X
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y = x ** self.c
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

# 미세한 차리를 이용하여 함수의 변화량을 구하는 방법을 '수치 미분(numercial differentiation)'이라고 한다.
# 아무리 작은 값을 사용하여도 오차가 발생 할 수 있음 -> '중앙 차분(centered difference)'을 통해 근사 오차를 줄임.

# f - 미분의 대상이 되는 함수
# x - 미분을 계산하는 병수, Variable
# eps - 작은 값, default = 1e-4
def num_diff(f, x, eps= 1e-4):
    x0 = v.Variable(x.data - eps)
    x1 = v.Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def comp_func(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

def square(x):
    # f = Square()
    # return f(x)
    return Square()(x)

def exp(x):
    # f = Exp()
    # return f(x)
    return Square()(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def no_grad():
    return c.using_config('enable_backprop', False)

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
    return Div()(x1, x0)
def pow(x, c):
    return Pow(c)(x)

def as_variable(obj):
    if isinstance(obj, v.Variable):
        return obj
    return v.Variable(obj)

if __name__ == '__main__':
    """
    # Function
    # print("Function 실습 코드")
    # x1 = v.Variable(np.array(10))
    # func1 = Function()
    # y1 = func1(x1)
    # print(type(y1))
    # print(y1.data)

    # Square
    print("Square 실습 코드")
    x2 = v.Variable(np.array(10))
    func2 = Square()
    y2 = func2(x2)
    print(type(y2))
    print(y2.data)
    print("\n")

    # 함수 연결 - 합성 함수 만들
    print("함수 연결 실습 코드")
    A = Square()
    B = Exp()
    C = Square()

    # x, a, b, y 모두 Variable 인스턴스
    # 다음과 같이 여러 함수를 순서대로 적용하여 만들어진 변환 전체를 합성 함수(composite function)라고 한다.
    x = v.Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    print(y.data)
    print("\n")

    # 수치 미분
    print("num_diff - 수치분 미 실습 코드")
    f = Square()
    x3 = v.Variable(np.array(2.0))
    dy3 = num_diff(f, x3)
    print(dy3)
    print("\n")

    # 합성 함수 미분
    print("num_diff with composite func")
    x4 = v.Variable(np.array(0.5))
    dy4 = num_diff(comp_func, x4)
    print(dy4)
    print("\n")

    # 역전파 (함수 연결 이용)
    print("backprop")
    y.grad = np.array(1.0)
    # b.grad = C.backward(y.grad)
    # a.grad = B.backward(b.grad)
    # x.grad = A.backward(a.grad)
    # print(x.grad)
    print("\n")

    # Backtrace
    print("backtrace")
    # 평가 결과가 true가 아니면 예외가 발
    assert y.creator == C
    assert y.creator.input == b
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x
    print("\n")

    # 역전파 도전
    print("backprop 2nd")
    # 작동 순
    # 1_함수를 가져온다.
    # 2_함수의 입력을 가져온다.
    # 3_함수의 backward 메서드를 호출한다.

    # C = y.creator
    # b = C.input
    # b.grad = C.backward(y.grad)
    #
    # B = b.creator
    # a = B.input
    # a.grad = B.backward(b.grad)
    #
    # A = a.creator
    # x = A.input
    # x.grad = A.backward(a.grad)
    #
    # print(x.grad)
    print("\n")

    # 역전파 자동화
    print("backprop auto")
    y.backward()
    print(x.grad)
    print("\n")
    
    # 가변길이 인수 적용
    print("Multiple inputs")
    xs = [v.Variable(np.array(2)), v.Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)
    print("\n")


    # 가변길이 인수 적용 - enhancement
    print("Multiple inputs - enhancement")
    x0 = v.Variable(np.array(2))
    x1 = v.Variable(np.array(3))
    # f = Add()
    # y = f(x0, x1)
    y = add(x0, x1)
    print(y.data)
    print("\n")

    # Square 클래스에도 가변길이 인수 적용
    print("Multiple inputs application on Square class")
    x13 = v.Variable(np.array(2.0))
    y13 = v.Variable(np.array(3.0))

    z13 = add(square(x13), square(y13))
    z13.backward()

    print(z13.data)
    print(x13.grad)
    print(y13.grad)
    print("\n")

    # 같은 변수 반복 사용
    print("Repetitive usage of same variables")
    x14 = v.Variable(np.array(3.0))
    y14 = add(add(x14, x14), x14)

    y14.backward()
    print('x.grad', x14.grad)
    print("\n")

    x143 = v.Variable(np.array(3.0))
    y143 = add(x143, x143)
    y143.backward()
    print(x143.grad)

    # 누적된 미분값 초기화
    # 28단계 - 로젠브록 함수 최적화(함수의 최솟값과 최댓값을 찾는 문제)
    x143.cleargrad()
    y143 = add(add(x143,x143),x143)
    y143.backward()
    print(x143.grad)

    x16 = v.Variable(np.array(2.0))
    a16 = square(x16)
    y16 = add(square(a16), square(a16))
    y16.backward()

    print(y16.data)
    print(x16.grad)

    for i in range(10):
        # 거대한 데이터
        x17 = v.Variable(np.random.randn(10000))
        # 복잡한 계산을 수행
        y17 = square(square(square(x17)))
        print(y17)

    x18 = v.Variable(np.array(1.0))
    x18_1 = v.Variable(np.array(1.0))
    t18 = add(x18, x18_1)
    y18 = add(t18, x18)
    y18.backward()

    print(y18.grad, t18.grad)
    print(x18.grad, x18_1.grad)

    print("With backprop")
    with c.using_config('enable_backprop', True):
        x182 = v.Variable(np.ones((100, 100, 100)))
        y182 = square(square(square(x182)))
        y182.backward()
        print(y182)
    print("\n")
    print("No backprop")
    # 중간 계산 결과 곧바로 삭제
    with c.using_config('enable_backprop', False):
        x183 = v.Variable(np.ones((100,100,100)))
        y183 = square(square(square(x183)))
        print(y183)
    with no_grad():
        x183 = v.Variable(np.ones((100,100,100)))
        y183 = square(square(square(x183)))
        print(y183)

    x19 = v.Variable(np.array([[1,2,3],[4,5,6]]))
    print(x19.shape)
    print(x19.ndim)
    print(x19.size)
    print(x19.dtype)
    print(len(x19))
    print(x19)

    a20 = v.Variable(np.array(3.0))
    b20 = v.Variable(np.array(2.0))
    c20 = v.Variable(np.array(1.0))

    # y20 = add(mul(a20,b20),c20)
    y20 = a20 * b20 + c20
    y20.backward()

    print(y20)
    print(a20)
    print(b20)
    print(c20)

    a201 = v.Variable(np.array(3.0))
    b201 = v.Variable(np.array(2.0))
    y201 = a201 * b201
    print(y201)
    """
    x21 = v.Variable(np.array(2.0))
    y21 = x21 + np.array(3.0)
    print(y21)

    x22 = v.Variable(np.array(2.0))
    y22_1 = 2.0 - x22
    y22_2 = x22 - 1.0
    y22_3 = x22 ** 3
    print(y22_1)
    print(y22_2)
    print(y22_3)
