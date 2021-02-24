import numpy as np

if __name__ == '__main__':
    x = np.array([[1,2,3],[4,5,6]])
    x2 = np.random.randn(1,2,3)
    y = np.sum(x, axis=(0), keepdims=True)
    print(y)
    y2 = np.sum(x2, axis=(0,1))
    print(x2)
    print("----")
    print(np.sum(x2, axis=(0)))
    print("----")
    print(np.sum(x2, axis=(1)))
    print("----")
    print(np.sum(x2, axis=(2)))
    print("----")
    print(y2)
