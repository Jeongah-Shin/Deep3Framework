import variable as v
import numpy as np

# Function 클래스는 기반 클래스로서 모든 함수에 공통되는 기능을 구현
# 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현

class Function:
    # input 은 Variable 인스턴스라고 가
    def __call__(self, input):
        # 데이터 꺼내기
        x = input.data
        y = self.forward(x)
        output = v.Variable(y)
        return output

    def forward(self, x):
        # 해당 메서드는 상속하여 구해야 한다.
        raise NotImplementedError()


class Square(Function):
    # Function 클래스를 상속하기 때문에
    # __call__ 메서드는 그대로 계승
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

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

if __name__ == '__main__':
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