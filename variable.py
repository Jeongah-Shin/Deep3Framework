import numpy as np
import pprint as pp

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
        # 세대 수를 기록하는 변수
        self.generation = 0
    def __str__(self):
        data = np.array_str(self.data)

        grad = None
        if self.grad is not None:
            grad = np.array_str(self.grad)

        creator = self.creator
        return '\n\tVaraible - data: {}, grad: {}, creator: {}'.format(data, grad, creator)
    def cleargrad(self):
        # 여러가지 미분을 연달아 계산할 때 똑같은 변수 재사용 가능
        self.grad = None
    def set_creator(self, func):
        self.creator = func
        # 부모세대 + 1로 세대를 기록한다.
        self.generation = func.generation + 1
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
            # 출력변수인 outputs에 담겨있는 미분값들을 리스트에 담기
            gys = [output.grad for output in f.outputs]
            # 역전파 호출하기
            gxs = f.backward(*gys)
            # gxs가 튜플이 아니라면 튜플로 변환
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            # 역전파로 전파되는 미분값을 Variable 인스턴스 변수인 grad에 저장
            # gxs와 f.inputs의 각 원소는 서로 대응 관계 - i번째 원소에 대해 f.inputs[i] 미분값은 gxs[i]에 대응

            # zip 함수는 동일한 개수로 이루어진 자료형을 묶어주는 역할
            # list(zip([1,2],[3,4])) ---> [(1,3),(2,4)]
            print('f.inputs ----> ', ' '.join(map(str, f.inputs)))
            print('gxs      ----> ', gxs)
            print('zipped results ', ['x: '+str(x).replace('\n','').replace('\t','') + ' / gx: ' + str(gx) for x, gx in zip(f.inputs, gxs)])
            print('\n')
            for x, gx in zip(f.inputs, gxs):
                # 같은 변수를 반복해서 사용하면 전파되는 미분 값이 덮어 써지는 현상 ---> 수정이 필요하다
                # x.grad = gx
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
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