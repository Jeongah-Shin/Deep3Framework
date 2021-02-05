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

### ➡️ Topology

그래프의 연결된 형태를 Topology(위상)이라고 한다. 다양한 위상의 계산 그래프 미분에 대응할 수 있도록 구조 변경!



복잡한 계산 그래프에서 함수의 우선순위를 정할 수 있도록 **Topology Sort(위상정렬)** 활용

→ 함수의 세대(generation)를 기록하기



### ➡️ Memory

- 파이썬은 필요 없어진 객체를 인터프리터가 메모리에서 자동으로 삭제한다.

  → 메모리 관리를 의식할 일이 크게 줄어든다.

- Memory Leak(메모리 누수)이나 Out of Memory(메모리 부족) 문제에 항상 대비해야 한다.



**👉 CPython의 메모리 관리 방식**

1. Reference(참조) 수를 세는 방식 → 참조 카운트
2. Generation(세대)을 기준으로 쓸모없어진 객체를 회수 → Garbage Collection(가비지 컬렉션)



**1️⃣ 참조 카운트 방식**

모든 객체는 참조 카운트가 0인 상태로 생성, ***다른 객체가 참조할 때마다 1씩 증가***

반대로, ***객체에 대한 참조가 끊길 때 마다 1만큼 감소***, 0이 되면 파이썬 인터프리터가 회수

- 참조 카운트가 증가하는 경우

  - 대입 연산자를 사용할 때
  - 함수에 인수로 전달할 때
  - 컨테이너 타입 객체(리스트, 튜플, 클래스 등)에 추가할 때

```python
class obj:
  pass

def f(x):
  print(x)

a = obj() # obj()에 의해 생성된 개체를 a 변수에 대입: 참조 카운트 1
f(a) # 함수에 전달: a가 인수로 전달되기 때문에 함수 안에서는 참조 카운트 2
# 함수 완료: 함수 범위를 벗어나오면 참조 카운트 1
a = None # 대입 해제: 참조를 끊으면(아무도 참조하지 않는 상태) 참조 카운트 0
```
```python
a = obj()
b = obj()
c = obj()
# a - 1, 변수 대입
# b - 1, 변수 대입
# c - 1, 변수 대입

a.b = b
b.c = c
# a - 1
# b - 2, a가 b를 참조
# c - 2, b가 c를 참조

a = b = c = None
# a - 0, 참조 끊음
# b - 1, 참조 끊음
# c - 1, 참조 끊음
```
>a의 참조카운트 0 → a 즉시 삭제 → a 삭제의 여파로 b의 참조 카운트가 1에서 0으로 감소 → b 삭제 → b삭제의 여파로 c의 참조 카운트가 1에서 0으로 감소 → c 삭제



**2️⃣ 순환 카운트 방식**

```python
a = obj()
b = obj()
c = obj()
# a - 1, 변수 대입
# b - 1, 변수 대입
# c - 1, 변수 대입

# a,b,c 세개의 객체가 원 모양을 이루며 서로가 서로를 참조
a.b = b
b.c = c
c.a = a
# b - 2, a가 b를 참조
# c - 2, b가 c를 참조
# a - 2, c가 a를 참조

a = b = c = None
# a - 1, 참조 끊음
# b - 1, 참조 끊음
# c - 1, 참조 끊음
```

> → `a = b = c = None` 을 실행하는 것으로 순환 참조의 참조 카운트가 0이 되지 않는다(메모리에서 삭제되지 않음), 하지만 셋 다 접근이 불가능 (불필요한 객체)
>
> 이에 대한 대안이 **"Generational Garbage Collector(세대별 가비지 컬렉션)"**
>
> - 메모리가 부족해지는 시점에서 파이썬 인터프리터에 의해 자동 호출
> - 명시적 호출 가능 (gc 모듈 임포트 후 `gc.collect()`)

`weakref` **모듈의 활용**

```python
import weakref
import numpy as np

a = np.array([1,2,3])
b = weakref.ref(a)

# 약한 참조
b
# 참조된 데이터에의 접근
b()

# 참조 카운트 방식에 따라 ndarray를 메모리에서 삭제
# b는 약한 참조이기 때문에 참조 카운트에 영향을 주지 않음 -> dead
a = None
```

> python 메모리 사용량을 측정하기 위해서는 memory profiler를 활용해볼 것!



### ➡️ Memory Saving Strategies

1. 불필요한 미분 결과 즉시 삭제

   중간 변수의 미분값을 제거 → 역전파시 사용하는 메모리양 줄이기

2. 불필요한 계산 생략

   역전파가 필요 없는 경우에 대한 처리

   > 학습(training) 시에는 미분값을 구해야하지만 추론(inference) 시에는 순전파만 하기 때문에 중간 계산 결과를 곧바로 버릴 수 있다.
   >
   > - 역전파시 노드를 따라가는 순서인 세대(generation)를 지정해주는 부분 제거
   > - 여러개의 계산들을 연결해주는 `output.ser_creator(self)` 부분 제거
   >
   > → 메모리 사용량 감소

   **`with` 문을 활용한 모드 전환 구현을 위해 `contextlib` 모듈 활용**

   ```python
   import contextlib
   
   @contextlib.contextmanager
   def config_test():
     # yield 전에 전처리 로직
     print('start')
     try:
       yield
     finally:
       # yield 다음에 후처리 로직
       print('done')
       
   with config_test():
     print('process...')
     
   """
   출력 결과
   start
   proces...
   done
   """
   ```

   > `with` 블록 안에서도 예외가 발생할 수 있고, 발생한 예외는 `yield`를 실행하는 코드로도 전달되므로 `try/finally`로 감싸야 한다.



### ➡️ Packaging

- **모듈** - 파이썬 파일

  다른 파이썬 프로그램에서 `import ` 하여 사용하는 것을 가정하고 만들어진 파이썬 파일

- **패키지** - 여러 모듈을 묶은 것

  패키지를 만들기 위해서는 디렉토리를 만들고 그 안에 모듈(파이썬 파일) 추가

- **라이브러리** - 여러 패키지를 묶은 것

  하나 이상의 디렉토리로 구성 (때로는 패키지를 가리켜 '라이브러리'라고 칭하기도 함)



### ➡️ Test functions for optimization

- **Sphere**

  단, 아래의 함수는 차원이 x, y, z 뿐인 3차원 공간에서의 Sphere 함수
$$
z = x^2 + y^2
$$

- **matyas (마차시)**
$$
 z = 0.26(x^2 + y^2) - 0.48xy
$$

- **Goldstein-Price**
$$
f(x, y) = [1 + (x+y+1)^2(19-14x+3x^2-14y+6xy+3y^2)]
 [30 + (2x-3y)^2(18-32x+12x^2+48y-36xy+27y^2)]
$$



✨ **Define-and-Run (정적 계산 그래프 방식)**

- 계산 그래프를 정의한 다음 데이터를 흘려보낸다.

  **👉 계산 그래프 정의 → 컴파일 → 데이터 흘려보내기**

  - 사용자 - 계산 그래프를 정의하여 제공함.
- 컴파일 단계에서 계산 그래프가 메모리상에 펼쳐짐.
  
  - 프레임워크 - 주어진 그래프를 컴퓨터가 처리할 수 있는 형태로 변환하여 데이터를 흘려보냄.


  pseudo - code 👇

 ```python
  # 가상의 Define-and-Run 방식 프레임워크용 코드 예시

  # 계산 그래프 정의
  # 실제 계산 X → 실제 '수치'가 아닌 '기호'
  # symbolic programming
  a = Variable('a')
  b = Varaible('b')
  c = a * b
  d = c + Constant(1)

  # 계산 그래프 컴파일
  f = compile(d)

  # 데이터 흘려보내기
  d = f(a=np.array(2), b=np.array(3))
 ```

  - 실제 데이터가 아닌 기호를 사용하여 추상적인 계산 절차를 코딩 + DSL(Domain-Specific Language) 사용
    - DSL(Domain-Specific Language) 란, 프레임워크 자체의 규칙들로 이루어진 언어를 의미

    ***ex)*** '상수는 Constant에 담아라'와 같은 규칙
    tensorflow의 if문 역할을 하는 `tf.cond` 👇
    
    ```python
    import tensorflow as tf
    
    flg = tf.placeholder(dtype=tf.bool)
    x0 = tf.placeholder(dtype=tf.float32)
    x1 = tf.placeholder(dtype=tf.float32)
    y = tf.cond(flg, lambda: x0 + x1, lambda: x0*x1)
    ```
    
- 대표적으로 tensorflow, Caffe, CNTK
  
  - tensorflow 2.0 부터는 Define-by-Run 방식도 차용

**정적 계산 그래프 방식의 장점**

- 성능 - 계산 그래프를 최적화하면 성능도 따라서 최적화

  - 계산 그래프의 최적화 → 계산 그래프의 구조와 사용되는 연산을 효율적인 것으로 변환

  ex) 두개의 연산을 하나로 축약 → 계산 시간 단축 (***a*b +1***의 연산 과정을 압축하여 ***add-mul***이라는 하나의 연산으로 표현)
  
  데이터를 흘려보내기 전에 전체 그래프가 손에 들어옴 → 계산 그래프 전체를 고려해 최적화 (for문에서 반복해 사용하는 연산을 하나로 축약)
  
- 어떻게 컴파일하느냐에 따라 다른 실행 파일로 변환 가능

  - 파이썬이 아닌 다른 환경에서도 데이터를 흘려보내는게 가능 → IoT 기기와 같은 edge 전용 환경에서 파이썬이 주는 오버헤드를 없앨 수 있음.

✨ **Define-by-Run (동적 계산 그래프 방식)**

- 데이터를 흘려보냄으로써 계산 그래프가 정의된다.

  - 2015년 체이너(Chainer)에 의해 처음 제창된 방법
  - '데이터 흘려보내기'와 '계산 그래프 구축'이 동시에 이루어짐.
  
- Deep3Framework 가 활용하는 방식
	```python
	import numpy as np
	from deep3framework import Variable
	
	a = Variable(np.ones(10))
	b = Variable(np.ones(10)*2)
	c = b * a
	d = c + 1
	print(d)
	```

- 대표적으로 PyTorch, MXNet, DyNet, tensorflow 2.0

**동적 계산 그래프 방식의 장점**

- 프레임워크 특유의 DSL을 배우지 않아도 된다.

- 계산 그래프를 '컴파일' 하여 독자적인 데이터 구조로 변환할 필요가 없다.

  → 일반 파이썬 프로그래밍으로 계산 그래프를 구축하고 실행할 수 있다.

  일반 파이썬의 if문이나 for문을 그대로 사용하여 계산 그래프를 만들 수 있다.
  
- 디버깅에의 유리하다.

  - 계산 그래프가 파이썬 프로그램 형태로 실행 → 디버깅도 파이썬 프로그램으로 가능 (pdb 활용)
  - 반대로 정적 계산 그래프에서 디버깅이 어려운 이유 - '계산 그래프 정의'와 '데이터 흘려보내기' 작업이 분리되어 있어서 어떤 과정이 문제가 되는지 특정하기 어려움.

#### Summary

|      | Define-and-Run(정적 계산 그래프)                             | Define-by-Run(동적 계산 그래프)                              |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 장점 | 1. 성능이 좋다.<br />2. 신경망 구조를 최적화하기 쉽다.<br />3. 분산 학습 시 더 편리하다. | 1. 파이썬으로 계산 그래프를 제어할 수 있다.<br />2. 디버깅이 쉽다.<br />3. 동적인 계산 처리에 알맞다. |
| 단점 | 1. 독자적인 언어(규칙)를 익혀야 한다.<br />2. 동적 계산 그래프를 만들기 어렵다.<br />3. 디버깅하기 매우 어려울 수 있다. | 1. 성능이 낮을 수 있다.                                      |

- 한 방식이 절대적이지 않기 때문에 두 모드를 모두 지원하는 프레임워크들이 많다.

  - PyTorch - 기본적으로 동적 계산 그래프 모드로 수행되지만 정적 계산 그래프 모드도 제공 (TorchScript 참고)

  - Tensorflow 도 2.0 버전부터 Eager Execution이라는 동적 계산 그래프 모드가 표준으로 채택, 필요시 정적 계산 그래프로 전환 가능

- 프로그래밍 언어 자체에서 자동 미분을 지원하려는 시도 → Swift for Tensorflow

  - 범용 프로그래밍 언어 Swift를 확장하여(스위프트 컴파일러를 손질하여) 자동 미분 구조를 도입하려는 시도

    → 성능과 사용성이라는 두 마리의 토끼를 모두 잡을 수 있다!

### ➡️ Higher order derivative

> '고계 도함수', '고계 미분' 이라고도 부르지만 지금부터 '고차 미분'이라는 용어로 통일

고차 미분이란 어떤 함수를 2번 이상 미분한 것 == 역전파에 대한 역전파

**Graphviz dot 파일 시동**

```python
dot dot_examples/sample.dot -T png -o dot_examples/sample.png
```

### ➡️ Taylor Series

Sin 함수의 미분을 Taylor Series(테일러 급수)를 통해 구해보기

> **n차 미분의 예시**
>
> - 1차 미분 : 위치의 미분(변화) == 속도
> - 2차 미분: 위치의 1차 미분 속도의 미분(변화) == 가속도

![T_{f}(x)=\sum _{{n=0}}^{\infty }{\frac  {f^{{(n)}}(a)}{n!}}\,(x-a)^{n}=f(a)+f'(a)(x-a)+{\frac  12}f''(a)(x-a)^{2}+{\frac  16}f'''(a)(x-a)^{3}+\cdots ](https://wikimedia.org/api/rest_v1/media/math/render/svg/436956d0b199f0699ddcd739eb4e421b45ca8133)

a = 0 일 때의 테일러 급수를 Maclaurin's series(매클로린 전개)라고 한다.



Threshold = 0.0001일 때 보다 Threshold = 1e-150 일 때의 계산 그래프 복잡성(깊이)이 기하급수적으로 증가했다.

→ 함수의 근사를 정확히 하기 위해 for 문의 반복횟수가 늘어났기 때문