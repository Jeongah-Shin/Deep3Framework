import numpy as np
from deep3framework import Variable
import deep3framework.functions as F


if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    t = x + x
    y = F.sum(t)

    # 각 변수의 미분값이 구해짐.
    # retain_grad = True 설정으로 미분값이 유지
    y.backward(retain_grad=True)
    # 기울기의 형상과 데이터(순전파 시 데이터)의 형상이 일치
    # x.shape == x.grad.shape
    print(y.grad)
    print(t.grad)
    print(x.grad)
    print(c.grad)