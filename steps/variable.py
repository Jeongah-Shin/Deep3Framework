import numpy as np
from steps import function as f


class Variable:
    # 연산시 좌항이 ndarray일 경우, ndarray의 __add__ 메소드가 호출되는 것을 막기 위해 priority를 높임.
    __array_priority__ = 200
    def __init__(self, data, name=None):
        # ndarray가 아닌 타입이 들어왔을 때에 대한 예외 처리
        if data is not None:
            if not isinstance(data, np.ndarray):
               raise TypeError('{}의 타입은 지원하지 않습니다.'.format(type(data)))
        # 아래의 두 값은 모두 numpy의 다차원 배열(ndarray)이라고 가정
        # 통상값 (data)
        self.data = data
        self.name = name
        # 미분값 (grad), 실제로 역전파를 하면 미분값을 계산하여 대입
        self.grad = None
        self.creator = None
        # 세대 수를 기록하는 변수
        self.generation = 0

    # 아래 notation으로 메서드를 인스턴스 변수처럼 사용할 수 있음
    @property
    def shape(self):
        return self.data.shape
    # 차원 수
    @property
    def ndim(self):
        return self.data.ndim
    # 원소 수
    @property
    def size(self):
        return self.data.size
    # 데이터 타입
    # dtype을 지정하지 않은 ndarray는 보통 float64 혹은 int64로 초기화
    # 신경망에서는 보통 float32 사용
    @property
    def dtype(self):
        return self.data.dtype
    def __len__(self):
        return len(self.data)
    # def __repr__(self):
    #     if self.data  is None:
    #         return 'variable(None)'
    #     p = str(self.data).replace('\n', '\n' + ' ' * 9)
    #     return 'variable(' + p + ')'
    def __str__(self):
        data = np.array_str(self.data).replace('\n', '')

        grad = None
        if self.grad is not None:
            if not isinstance(data, np.ndarray):
                grad_arr = np.array(self.grad)
                grad = np.array_str(grad_arr)
            else:
                grad = np.array_str(self.grad)

        creator = self.creator
        return '\n\tVaraible - data: {}, grad: {}, creator: {}'.format(data, grad, creator)
    """
    Variable.__add__ = f.add
    Variable.__radd__= f.add
    Variable.__mul__= f.mul
    Variable.__rmul__= f.mul
    를 Variable class 바깥에 선언해줌으로써 아래의 메서드들의 선언을 대체할 수 있음.
    """
    # 피연산자가 좌항일 때
    # def __mul__(self, other):
    #     return f.mul(self, other)
    # 피연산자가 우항일 때
    # def __rmul__(self, other):
    #     return f.mul(self, other)
    # def __add__(self, other):
    #     return f.add(self, other)
    # def __radd__(self, other):
    #     return f.add(self, other)
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
    def backward(self, retain_grad=False):
        if self.grad is None:
            # self.data 와 같은 형상으로 ndarray 인스턴스 생성
            # 모든 요소를 1로 채워서 돌려줌.
            # self.data가 스칼라이면 self.grad도 스칼라!
            self.grad = np.ones_like(self.data)

        # funcs = [self.creator]
        funcs = []
        # funcs 리스트에 같은 함수를 중복 추가하는 일을 막기 위해 set 사용
        seen_set = set()
        # 함수 리스트를 세대 순으로 정렬하는 역할
        # funcs.pop()은 자동으로 세대가 가장 큰 함수를 꺼내게 된다.
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # 리스트의 원소를 x라고 했을 때 x.generation의 값을 키로 사용하여 정렬
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

        while funcs:
            # 1_함수를 가져온다.
            f = funcs.pop()

            # 2_ 함수의 입력과 출력을 가져온다.
            # 출력변수인 outputs에 담겨있는 미분값들을 리스트에 담기
            # output() -> 약한 참조의 값을 가져오기 위해서 사용
            gys = [output().grad for output in f.outputs]
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
                    # funcs.append(x.creator)
                    add_func(x.creator)
            # retain_grad가 True 이면, 모든 변수가 미분 결과(기울기) 유지
            # retain_grad가 False 이면, 중간 변수의 미분값을 None으로 재설정
            if not retain_grad:
                for y in f.outputs:
                    # y는 약한 참조
                    y().grad = None

Variable.__add__ = f.add
Variable.__radd__= f.add
Variable.__mul__= f.mul
Variable.__rmul__= f.mul
Variable.__neg__ = f.neg
Variable.__sub__ = f.sub
Variable.__rsub__ = f.rsub
Variable.__truediv__ = f.div
Variable.__rtruediv__ = f.rdiv
Variable.__pow__ = f.pow

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