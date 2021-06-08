import numpy as np
import matplotlib.pyplot as plt

d=2; n=1000
x=np.matrix(np.random.randn(n,d))
s=np.matrix([[1., 0.],[0., 2.]])
r=np.matrix([[np.cos(np.pi/3), -np.sin(np.pi/3)],[np.sin(np.pi/3), np.cos(np.pi/3)]])
t=np.matrix([0.5, -1.])
x=x.dot(s).dot(r)+np.ones([n,1]).dot(t)
m=np.mean(x,axis=0);
sig=(x-np.ones([n,1]).dot(m)).T.dot(x-np.ones([n,1]).dot(m))/n
isig=np.linalg.inv(sig)
xx,yy=np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
xt=xx-m[0,0]
yt=yy-m[0,1]
p=1./(2.*np.pi*np.sqrt(np.linalg.det(sig))) * np.exp(-1./2.*(isig[0,0]*xt*xt+(isig[0,1]+isig[1,0])*xt*yt+isig[1,1]*yt*yt))
plt.figure()
plt.scatter([x[:,0]],[x[:,1]])
plt.contour(xx,yy,p,cmap='hsv')
plt.show()
