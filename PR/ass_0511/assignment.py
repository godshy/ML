from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# データの読み込み
data = loadmat('data20210511.mat')
x1 = data['x1']
x2 = data['x2']
#print(x1.shape)

def pc(p):  # P(x)を求める
    result = 0
    for i in range(500):
        result += 1./2. * p[i]
    return result




def gauss_kernel():
    h1l = np.array([1., 0.5, 0.1])  # hnが異なる場合を定義する
    x = np.linspace(-3., 3., num=500)  # xを定義する
    f = lambda x, xi, hn: 1./hn*1./np.sqrt(2.*np.pi)*np.exp(-((x-xi)/hn)**2./2.)  #　gaussian カーネル関数を定義する
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(len(h1l)):   # hnが異なる場合
        h1 = h1l[i]
        hn = h1/np.sqrt(200)
        p = np.zeros(len(x))
        x_i = 0
        for x_i in range(len(x2.T)):    # x2のデータをx_iにする, この部分のx2をx1で書き換えると、x1の結果が出る
            p = p + f(x, x2.T[x_i], hn)
        p = p / 200
        pc1 = pc(p)
        pci = 1./2. * p/pc1
        plt.subplot(1, len(h1l), i+1, title="h1="+str(h1l[i]))  #　図を描く
        plt.plot(x,p, color = 'red')
        plt.plot(x, pci)
#        print(x.shape,p.shape)
    plt.show()




def ker(k):
    sum = 0
    for i in range(500):    # カーネル関数の定義
        if k[i] <=0.5:
            k[i] = 1
        else:
            k[i] = 0
    return k




def rectan_kernel():
    h1l = np.array([1., 0.5, 0.1])
    x = np.linspace(-3., 3., num=500)
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    for i in range(3):
        p = np.zeros(len(x))
        for x_i in range(len(x2.T)):
            k = (x - x2.T[x_i])/h1l[i]   # kernel関数
            b = ker(k)
            p = p + b     # sum
        p = (1/(200*0.5**500)) * p  # pn(x)の計算

        plt.subplot(1, len(h1l), i+1, title="h1="+str(h1l[i]))
        plt.plot(x, p)
        pc1 = pc(p)
        pci = 1./2. * p/pc1
        plt.plot(x, pci)
    plt.show()

gauss_kernel()
rectan_kernel()





