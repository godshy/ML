import numpy as np
import matplotlib.pyplot as plt

# nl: list of n (num. of points)
nl=np.array([1, 10, 100, 100000])
# h1l: list of h1 (size of Parzen window)
h1l=np.array([1., 0.5, 0.1])
x=np.linspace(-3.,3.,num=500)
# assuming Gaussian window
f=lambda x,xi,hn: 1./hn*1./np.sqrt(2.*np.pi)*np.exp(-((x-xi)/hn)**2./2.)
# if you want to use box function, use the following definition
#f=lambda x,xi,hn: 1./hn*(np.abs(x-xi)/hn<=0.5)
plt.figure(figsize=(10, 10))
plt.subplots_adjust(wspace=0.2, hspace=0.5)
for i in range(len(h1l)):
  h1=h1l[i]
  for j in range(len(nl)):
    n=nl[j]
    hn=h1/np.sqrt(n)
    p=np.zeros(len(x))
    # for each point out of n points yielding Gaussian...
    for xi in np.random.randn(n):
      p=p+f(x,xi,hn)
#     print(p)

    p=p/n
    print(p.shape)
    plt.subplot(len(nl),len(h1l),i+j*len(h1l)+1,title="h1="+str(h1l[i])+" n="+str(nl[j]))
    plt.plot(x,p)
plt.show()
