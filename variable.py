import numpy as np

class Variable:
    def __init__(self, data):
        # ndarray가 아닌 타입이 들어왔을 때에 대한 예외 처리
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}의 타입은 지원하지 않습니다.'.format(type(data)))
        # 아래의 두 값은 모두 numpy의 다차원 배열(ndarray)이라고 가정
        # 통상값 (data)
        self.data = data
        # 미분값 (grad), 실제로 역전파를 하면 미분값을 계산하여 대입
        self.grad = None
        self.creator = None
    def set_creator(self, func):
        self.creator = func
    # 재귀를 이용한 구현
    def backward_recursion(self):
        # 1_함수를 가져온다.
        f = self.creator
        # creator가 없으면 역전파 중단
        # creator가 없다는 것은 해당 Variable은 함수 바깥에서 생성 되었음(사용자 력)을 의미
        if f is not None:
            # 2_함수의 입력을 가져온다.
            x = f.input
            # 3_함수의 backward 메서드를 호출한다.
            x.grad = f.backward(self.grad)
            # 하나 앞 변수의 backward 메서드를 호출한다(재귀)
            x.backward()
    # 반복문을 이용한 구현
    def backward(self):
        if self.grad is None:
            # self.data 와 같은 형상으로 ndarray 인스턴스 생성
            # 모든 요소를 1로 채워서 돌려줌.
            # self.data가 스칼라이면 self.grad도 스칼라!
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            # 1_함수를 가져온다.
            f = funcs.pop()
            # 2_ 함수의 입력과 출력을 가져온다.
            x,y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                # 하나 앞의 함수를 리스트에 추가한다.
                funcs.append(x.creator)

if __name__ == '__main__':
    data = np.array(1.0)
    x = Variable(data)

    x.data = np.array(2.0)
    print(x.data)

    # Scalar - 0 dimensional Tensor
    a = np.array(1)
    print(a.ndim)

    # Vector - 1 dimensional Tensor
    b = np.array([1,2,3])
    print(b.ndim)

    # Matrix - 2 dimensional Tensor
    c = np.array([[1,2,3],
                  [4,5,6]])
    print(c.ndim)