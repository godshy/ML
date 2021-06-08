import numpy as np
import matplotlib.pyplot as plt

# nl: list of num. of points
nl=np.array([1, 16, 256, 100000])
# h1l: list of the size of Parzen window
h1l=np.array([1., 0.5, 0.1])
x=np.linspace(0.,4.,num=500)
# f: Gaussian kernel or box kernel
f=lambda x,xi,hn: 1./hn*1./np.sqrt(2*np.pi)*np.exp(-((x-xi)/hn)**2./2.)
#f=lambda x,xi,hn: 1./hn*(np.abs(x-xi)/hn<=0.5)
# g: complicated, huh?
# inverse function of cumulative dist. func. (CDF) of triangle and box.
# this is needed to generate "random" points yielding triangle and box dist.
g=lambda x: np.where((0.<=x) and (x<1./3.),np.sqrt(3.*x)+0.5,\
            np.where((1./3.<=x) and (x<2./3.), 2.5-np.sqrt(2.-3.*x),\
            np.where((2./3.<=x),3./2.*x+2., 0)))
plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
for i in range(len(h1l)):
  h1=h1l[i]
  for j in range(len(nl)):
    n=nl[j]
    hn=h1/np.sqrt(n)
    p=np.zeros(len(x))
    for xi in np.random.rand(n):
      p=p+f(x,g(xi),hn)
    p=p/n
    plt.subplot(len(nl),len(h1l),i+j*len(h1l)+1,title="h1="+str(h1l[i])+" n="+str(nl[j]))
    plt.plot(x,p)
plt.show()
