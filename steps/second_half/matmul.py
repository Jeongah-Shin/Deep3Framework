from deep3framework import Variable
import deep3framework.functions as F
import numpy as np

x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
y.backward()

print(x)
print(W)
print(x.grad.shape)
print(W.grad.shape)
print(y.shape)