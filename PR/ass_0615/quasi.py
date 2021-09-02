import numpy as np
import matplotlib.pyplot as plt

d=2
n=100
x=2*np.random.rand(d,n)-1
l=2*((2*x[0,:]+x[1,:])>0.5)-1
flip=abs((2*x[0,:]+x[1,:])-0.5)<0.2
l[np.where(flip)]=-l[np.where(flip)]

plt.figure()
plt.plot(x[0,np.where(l==1)],x[1,np.where(l==1)],'bo')
plt.plot(x[0,np.where(l==-1)],x[1,np.where(l==-1)],'bx')
plt.show()