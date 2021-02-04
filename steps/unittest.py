import unittest
import numpy as np
from steps import function as f, variable as v


class SquareTest(unittest.TestCase):
    # python -m unittest unittest.py
    # 아니면 unit test를 수행하고 싶은 클래스에서 test.unittest.main()
    def test_forward(self):
        x = v.Variable(np.array(2.0))
        y = f.square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)
    def test_backward(self):
        x = v.Variable(np.array(3.0))
        y = f.square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    def test_grad_check(self):
        x = v.Variable(np.random.rand(1))
        y = f.square(x)
        y.backward()
        num_grad = f.num_diff(f.square, x)
        # 두 메서드로 각각 구한 값들이 거의 일치하는지 확인
        # np.allclose(a,b) ---> ndarray 인스터인 a와 b의 값이 가까운지 판정
        # 얼마나 가까운지 표현하려면 rtol, atol 인수 부여
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)