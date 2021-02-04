# 본 파일이 위치한 디렉터리의 부모 디렉터리(..)를 모듈 검색 경로에 추가
# 파이썬 명령어를 어디에서 실행하든 dezero 디렉터리의 파일들을 제대로 임포트할 수 있음.
# deep3framework가 패키지로 설치된 경우라면 필요 없으나 개발 중인 디렉터리를 import 하기 위해 임시적으로 사용
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from deep3framework import Variable

x = Variable(np.array(1.0))
y = (x+3) ** 2
y.backward()

print(y)
print(x.grad)

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

x_s = Variable(np.array(1.0))
y_s = Variable(np.array(1.0))
z_s = sphere(x_s, y_s)
z_s.backward()
print(x_s.grad, y_s.grad)

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

x_m = Variable(np.array(1.0))
y_m = Variable(np.array(1.0))
z_m = matyas(x_m, y_m)
z_m.backward()
print(x_m.grad, y_m.grad)

def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

x_g = Variable(np.array(1.0))
y_g = Variable(np.array(1.0))
z_g = goldstein(x_g, y_g)
z_g.backward()
print(x_g.grad, y_g.grad)