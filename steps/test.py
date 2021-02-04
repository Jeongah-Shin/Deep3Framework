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