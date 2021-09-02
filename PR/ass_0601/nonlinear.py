import numpy as np
import matplotlib.pyplot as plt

d=2
n=100
x=2*np.random.rand(d,n)-np.array([1,1])[:,np.newaxis]
l=2*(((2*x[0,:]+x[1,:])>0.5) != ((x[0,:]-1.5*x[1,:])>0.5))-1
"""
plt.figure()
plt.plot(x[0,np.where(l==1)],x[1,np.where(l==1)],'bo')
plt.plot(x[0,np.where(l==-1)],x[1,np.where(l==-1)],'bx')
plt.show()
"""
