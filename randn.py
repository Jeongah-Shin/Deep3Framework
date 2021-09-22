import numpy as np
import matplotlib.pyplot as plt

dy = np.random.rand(2, 3)
yh = np.random.rand(2, 3)

temp_ = np.random.rand(2, 3)

dy_dot = dy * temp_
yh_dot = yh * temp_

if dy_dot.max() > yh_dot.max():
    print("Welcome Daeyun to Goin Water Team !")
else :
    print("Welcome Yonghyun to Goin Water Team !")