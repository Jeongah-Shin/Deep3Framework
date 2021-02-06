import numpy as np
from deep3framework import Variable
from deep3framework.utils import get_dot_graph
from deep3framework.utils import plot_dot_graph
import steps.test as t
import os


if __name__ == '__main__':
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y1 = x0 + x1

    x0.name = 'x0'
    x1.name = 'x1'
    y1.name = 'y'

    txt = get_dot_graph(y1, verbose=False)
    print(txt)

    path = os.getcwd() + '/dot_examples/sample2.dot'
    with open(path, 'a+') as o:
        o.write(txt)

    x2 = Variable(np.array(1.0))
    y2 = Variable(np.array(1.0))
    z2 = t.goldstein(x2, y2)
    z2.backward()

    x2.name = 'x2'
    y2.name = 'y2'
    z2.name = 'z2'
    plot_dot_graph(z2, verbose=False, to_file='dot_examples/goldstein.png')