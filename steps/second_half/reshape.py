import numpy as np
from deep3framework import Variable
import deep3framework.functions as F

if __name__ == '__main__':
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    # 아래 과정에서 y의 기울기도 자동으로 채워진다.
    # 단, y.grad.shape == y.data.shape
    y.backward(retain_grad=True)
    print(y)
    print(x.grad)

    print("\n\n")
    # 0 ~1 사이의 균일 분호 표준정규분포 난수를
    # matrix array(1, 2, 3)의 형태로 생성
    x_np = np.random.rand(1, 2, 3)
    print("x_np - " + str(x_np))

    y_np1 = x_np.reshape((2, 3))
    print("Tuple - " + str(y_np1))
    y_np2 = x_np.reshape([2, 3])
    print("List - " + str(y_np2))
    y_np3 = x_np.reshape(2, 3)
    print("params itselves - " + str(y_np3))

    x_d = Variable(np.random.randn(1, 2, 3))
    # 튜플
    y_t = x_d.reshape((2,3))
    print(y_t)
    # 가변 길이 인수
    y_d = x_d.reshape(2,3)

