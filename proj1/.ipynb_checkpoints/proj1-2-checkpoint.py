
#计算累积分布函数
#累积分布函数是概率密度函数的积分，表示随机变量X的概率分布
#小于随机变量某个值概率的和

import numpy as np
import matplotlib.pyplot  as plt 
import math
import numpy.matlib
np.random.seed(0)
N = np.matlib.randn(10000, 1)


def normfun(x, miu, delta):

    pdf = np.exp(-((x - miu)**2) / (2* delta**2)) / (delta * np.sqrt( 2 * np.pi))
    return pdf

miu = 0
delta = 1
X = np.arange(-3, 3, 0.1)
Y = normfun(X, miu, delta)
Cy = np.cumsum(Y*0.1)
plt.plot(X, Y)
plt.plot(X, Cy, 'r--')
plt.hist(N, bins = 6, color = 'g', rwidth = 0.9, density = True) #normed为density
plt.xlabel('random value')
plt.ylabel('pdf')
