
#生成正态分布函数的图表
#標本値から確率密度関数を求める
#正态分布中x取a-b之间概率有多大？
#就是ab之间曲线与x轴围成曲边梯形的面积

import numpy as np
import matplotlib.pyplot  as plt 
import math
import numpy.matlib
np.random.seed(0)
N = np.matlib.randn(10000, 1)    #标本值


def normfun(x, miu, delta):

    pdf = np.exp(-((x - miu)**2) / (2* delta**2)) / (delta * np.sqrt( 2 * np.pi))
    return pdf

miu = 0
delta = 1
X = np.arange(-3, 3, 0.1)
Y = normfun(X, miu, delta)
plt.plot(X, Y)
plt.hist(N, bins = 6, color = 'g', rwidth = 0.9, density = True) #normed为density
plt.xlabel('random value')
plt.ylabel('pdf')
