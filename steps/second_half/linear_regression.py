import numpy as np
from deep3framework import Variable
import deep3framework.functions as F

# y = Wx + b
if __name__ == '__main__':
    np.random.seed(0) # 시드값 고정 (같은 데이터를 나중에 재현해야 하므로)
    x = np.random.rand(100, 1)
    y = 5 + 2*x + np.random.rand(100, 1) # y에 무작위 노이즈 추가
    print(x)
    print("--------------------------------------------------")
    print(y)
    x, y = Variable(x), Variable(y)

    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = F.matmul(x, W) + b
        return y

    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return F.sum(diff ** 2) / len(diff)

    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        print(W, b, loss)
