from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat('data20210511.mat')
x1 = data['x1']
x2 = data['x2']


def pc(p):
    result = 0
    for i in range(500):
        result += 1./2. * p[i]
    return result



nl = np.array([1, 16, 256, 32768])   # 様々な k
x = np.linspace(-3., 3., num=500)
plt.figure(figsize=(8, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
for i in range(len(nl)):
    n = nl[i]
    k = np.sqrt(n)  #固定の k
    p = np.zeros(500)
    for j in range(500):
        r = sorted(abs(x1.T-x[j]))  # x1に書き換える可能
        p[j] = k/(n*2.*r[int(k)-1])  # pn(x) の計算
    ax = plt.subplot(2, 2, i+1, title="n="+str(nl[i]))
 #   print(x.shape,  p.shape)
    pc1 = pc(p)
    pci = 1./2. * p/pc1
    plt.plot(x,p, color = 'red')
    plt.plot(x, pci)
    if k==1:
        ax.set_ylim([0, 10])

plt.show()
