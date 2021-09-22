import deep3framework.layers as L
import deep3framework.functions as F
from deep3framework import Layer

model = Layer()
model.l1 = L.Linear(5) # output 크
model.l2 = L.Linear(3)

def predict(model, x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

# 모델의 모든 매개변수에의 접근
for p in model.params():
    print(p)

# 모든 매개변수의 기울기를 재설정
model.cleargrads()