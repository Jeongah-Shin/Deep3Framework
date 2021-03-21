import numpy as np
from deep3framework import Variable
import deep3framework.functions as F
import deep3framework.layers as L

if __name__ == '__main__':
    # 데이터셋 준비
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)  # 데이터 생성에 sin 함수 이용비

    # # 가중치 초기화
    # I, H, O = 1, 10, 1
    # W1 = Variable(0.01 * np.random.randn(I, H))
    # b1 = Variable(np.zeros(H))
    # W2 = Variable(0.01 * np.random.randn(H, O))
    # b2 = Variable(np.zeros(0))

    l1 = L.Linear(10) # 출력 크기 지정
    l2 = L.Linear(1)

    # 신경망 추론
    def predict(x):
        # y = F.linear(x, W1, b1)
        y = l1(x)
        y = F.sigmoid(y)
        # y = F.linear(y, W2, b2)
        y = l2(y)
        return y

    lr = 0.2
    iters = 10000

    # 신경망 학습
    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        # W1.cleargrad()
        # b1.cleargrad()
        # W2.cleargrad()
        # b2.cleargrad()
        l1.cleargrads()
        l2.cleargrads()
        loss.backward()

        # W1.data -= lr * W1.grad.data
        # b1.data -= lr * b1.grad.data
        # W2.data -= lr * W2.grad.data
        # b2.data -= lr * b2.grad.data

        for l in [l1, l2]:
            for p in l.params():
                p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)