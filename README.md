# 🐥 밑바닥부터 시작하는 딥러닝3 실습



## 🔑 Keywords



### ➡️ Variable

### ➡️ Fucntion

### ➡️ Numercial Differentiation

- 미세한 차리를 이용하여 함수의 변화량을 구하는 방법을 '수치 미분(numercial differentiation)'이라고 한다.

- 아무리 작은 값을 사용하여도 오차가 발생 할 수 있다.

  → x-h, x+h 에서의 기울기를 구하는 방법 **'중앙 차분(centered difference)'**

  → x, x+h 지점에서의 기울기를 구하는 방법 **'전진 차분(forward difference)'**

  중앙 차분이 오차가 더 적다!

**💦 수치 미분의 문제점**

1. 자릿수 누락 → 차이를 구하는 계산에서 주로 크기가 비슷한 값들을 다루므로 계산 결과에서 유효 자릿수가 줄어들 수 있음.

> **ex)** 1.234... - 1.233... = 0.001434...
>
> 유효자릿수가 4일 때 1.234 - 1.233 = 0.001 ***(유효 자릿수가 1로 줄어듦)***

2. 계산량이 많음 → 신경망에서는 매개변수가 수백만 개 이상 사용하는 경우도 많음.

✨ 위의 한계를 극복한 **역전파**, 역전파를 정확하게 구현했는지 확인하기 위해 수치 미분의 결과를 이용

### ➡️ Backpropagation

**Chain Rule (연쇄 법칙)**

- 합성함수의 미분은 구성 함수 각각을 미분한 후 곱한 것과 같음.
  - 전파되는 값은 최종 결과인 y값의 미분값들

**Loss Function (손실함수)**

- 대개 머신러닝 모델은 대량의 매개변수를 입력받아서 마지막에 손실함수를 거쳐 출력을 내는 형태로 진행

  - (많은 경우) 단일 스칼라값

- 손실 함수의 각 매개변수에 대한 미분을 계산해야함 → 출력에서 입력 방향으로 한번만 전파하면 모든 매개변수에 대한 미분을 계산할 수 있음.

**Gradient (기울기)**

벡터나 행렬 등 다변수에 대한 미분

> **Define-by-Run (동적 계산 그래프)**
>
> 데이터를 흘려보냄으로써(Run함으로써) 연결이 규정된다(Define 된다)
>
> → 딥러닝에서 수행하는 계산들을 계산 시점에 '연결' 하는 방식 (체이너와 파이토치의 방식)
>
> → 분기가 있는 계산 그래프에의 응용을 위한 웬거트 리스트( Wengert List)

### ➡️ Python Unit Test

```python
python -m unittest test.py

# 특정 디렉토리 아래 모두를 실행하고 싶으면
python -m unittest discover <dir_name>
```

**✨ 컴퓨터 프로그램에서 미분을 계산하는 방법**

- Numercial Differentiation (수치 미분)

  변수에 미세한 차이를 주어 일반적인 계산(순전파) 2회 시행 후 출력의 차이로부터 근사적으로 미분을 계산, 다량의 변수를 사용하느 함수를 다룰 때의 계산 비용이 높음.

- Symbolic Differentiation (기호 미분)

  공식으로 계산, 최적화를 고혀하지 않고 구현하면 수식이 곧바로 거대해질 우려가 있음.

- Automatic Differentiation (자동 미분)

  연쇄 법칙의 사용

  - forward mode
  - reverse mode == 역전파

### ➡️ Partial Derivative

Partial Derivative(편미분)은 입력 변수가 여러개인 다변수 함수에서 하나의 입력 변수에만 주목하여(다른 변수는 상수로 취급) 미분하는 것을 뜻함.
$$
{\partial y \over \partial x_0} = 1, {\partial y \over \partial x_1} =1
$$
