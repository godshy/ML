import numpy as np
import matplotlib.pyplot as plt

nl=np.array([1, 16, 256, 65536])
x=np.linspace(0,4,num=500)
g=lambda x: np.where((0.<=x) and (x<1./3.),np.sqrt(3.*x)+0.5,\
            np.where((1./3.<=x) and (x<2./3.), 2.5-np.sqrt(2.-3.*x),\
            np.where((2./3.<=x),3./2.*x+2., 0)))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
for i in range(len(nl)):
  n=nl[i]
  k=np.sqrt(n);
  s=list(map(g,np.random.rand(n)))
  p=np.zeros(len(x))
  for j in range(len(x)):
    r=sorted(abs(s-x[j]))
    p[j]=k/(n*2.*r[int(k)-1])
  ax=plt.subplot(2,2,i+1,title="n="+str(nl[i]))
  plt.plot(x,p)
  if k==1:
    ax.set_ylim([0, 10])
plt.show()
