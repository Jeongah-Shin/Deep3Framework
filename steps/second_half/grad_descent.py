import numpy as np
from deep3framework import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2)**2 + (1-x0)**2
    return y

if __name__ == '__main__':
    print("rosenbrock function call\n")
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    y = rosenbrock(x0, x1)
    y.backward()
    print(x0.grad, x1.grad)
    # 출력값 -> -2.0, 400.0
    # (x0, x1) = (0.0, 2.0) 일 때,
    # y 값을 가장 크게 늘려주는 방향이 (-2.0, 400.0)
    # (마이너스를 곱한) y값을 가장 작게 줄여주는 방향이 (2.0, -400.0)


    # 로젠브록 함수의 최솟값 찾기
    print("gradient descent approach to rosenbrock func\n")
    lr = 0.001 # learning rate
    iters = 10000 # 반복 횟수

    for i in range(iters):
        print("{}번쨰 iteration\n\tx0: {}\n\tx1: {}".format(i, x0, x1))

        y = rosenbrock(x0, x1)

        # x0, x1에 grad에는 미분값이 계속 누적되기 때문에
        # 새롭게 미분할 때는 지금까지 누적된 값을 초기화
        x0.cleargrad()
        x1.cleargrad()
        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

