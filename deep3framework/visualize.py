import numpy as np
from deep3framework import Variable
from deep3framework.utils import get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)

with open('examples/sample2.dot', 'w') as o:
    o.write(txt)