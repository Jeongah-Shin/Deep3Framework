import numpy as np
from deep3framework import Variable
from deep3framework.utils import plot_dot_graph
import deep3framework.functions as F


if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)

    iters = 0

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    # Visualizer

    gx = x.grad
    gx.name = 'gx' + str(iters + 1)
    plot_dot_graph(gx, verbose=False, to_file='images/tanh.png')