import numpy as np
import matplotlib.pyplot as plt

d=2
n=500
x=np.concatenate((np.random.randn(d,int(n/2))/4+np.array([0.5,0.5])[:,np.newaxis],\
                  np.random.randn(d,int(n/2))/4-np.array([0.5,0.5])[:,np.newaxis]),\
                  axis=1)
l=2*(x[0,:]>0)-1
"""
plt.figure()
plt.plot(x[0,np.where(l==1)],x[1,np.where(l==1)],'bo')
plt.plot(x[0,np.where(l==-1)],x[1,np.where(l==-1)],'bx')
plt.show()
"""
