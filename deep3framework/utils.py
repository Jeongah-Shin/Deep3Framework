import os
import subprocess
from deep3framework import Variable
import numpy as np

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    # id 함수는 주어진 객체의 ID를 반환 (객체 ID는 다른 객체와 중복되지 않기 때문의 노드의 ID로 사용하기 적합)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        # y는 약한 참조(weakref)
        txt += dot_edge.format(id(f), id(y()))
    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            funcs.sort(key=lambda x:x.generation)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose =True, to_file='f.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 1. dot 데이터를 파일에 저장
    # '~' 을 현재 사용자 홈 디렉토리의 절대경로로 대체
    tmp_dir = os.path.join(os.path.expanduser('~'), '.deep3frame')
    if not os.path.exists(tmp_dir):
        # ~/.deep3frame 디렉터리가 없다면 새로 생성
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # 2. dot 명령어 호출
    extension = os.path.splitext(to_file)[1][1:] # 확장자(png, pdf 등)
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    # 파이썬에서 외부 프로그램을 호출하기 위해 subprocess.run() 함수 사용
    subprocess.run(cmd, shell=True)


if __name__ == '__main__':
    print("variable dot\n")
    x = Variable(np.random.randn(2, 3))
    x.name = 'x'
    print(_dot_var(x))
    # verbose가 True일 때, ndarray 인스턴스의 shape과 type을 함께 반환
    print(_dot_var(x, verbose=True))

    print("function dot\n")
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    y = x0 + x1
    txt = _dot_func(y.creator)
    print(txt)