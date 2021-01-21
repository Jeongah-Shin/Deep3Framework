import variable as v
import numpy as np
import test as t

# Function 클래스는 기반 클래스로서 모든 함수에 공통되는 기능을 구현
# 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현

class Function:
    # input 은 Variable 인스턴스라고 가정
    def __call__(self, input):
        # 데이터 꺼내기
        x = input.data
        y = self.forward(x)
        # 0차원의 x = ndarray - np.array(1.0)
        # x ** 2를 하면 np.float64 or np.float32로 리턴하는 문제 해결
        output = v.Variable(as_array(y))
        # 출력 변수에 창조자를 설정
        output.set_creator(self)
        # 입력 변수를 보관해놨다가 역전파시 사용
        self.input = input
        # 출력도 저장
        self.output = output
        return output

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

class Square(Function):
    # Function 클래스를 상속하기 때문에
    # __call__ 메서드는 그대로 계승
    def forward(self, x):
        y = x ** 2
        return y
    # gy 는 ndarray 인스턴스
    def backward(self, gy):
        x = self.input.data
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
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

def brief_square(x):
    return Square()(x)

def brief_exp(x):
    return Exp()(x)

if __name__ == '__main__':
    # Unit Test 실행
    # t.unittest.main()

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
    #b
    # print(x.grad)
    print("\n")

    # 역전파 자동화
    print("backprop auto")
    y.backward()
    print(x.grad)
    print("\n")

    # 함수를 편리하게!
    print("func enhancement")
    x9 = v.Variable(np.array(0.5))
    # Type Error 테스트
    # x9 = v.Variable(0.5)
    # a9 = square(x9)
    # b9 = exp(a9)
    # y9 = square(b9)

    # 연속해서 적용하기
    y9 = square(exp(square(x9)))

    # np.ones_like()으로 채웠으므로 생략 가능
    # y9.grad = np.array(1.0)
    y9.backward()

    print(x9.grad)
    print("\n")