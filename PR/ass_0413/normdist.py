import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# 1D Gaussian
plt.figure()
xx=np.linspace(-3,3)
plt.plot(xx,1./np.sqrt(2.*np.pi)*np.exp(-xx**2./2));
# plt.savefig('normdist1d.eps')

# 2D Gaussian
s=np.matrix([[1., 0.], [0., 2.]])
th=np.pi/3.
r=np.matrix([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
sig=(s.dot(r)).T.dot(s.dot(r));
isig=np.linalg.inv(sig)
xx,yy=np.meshgrid(np.linspace(-5,5),np.linspace(-5,5))
p=1./(2.*np.pi*np.sqrt(np.linalg.det(sig))) * \
    np.exp(-1./2.*(isig[0,0]*xx*xx+(isig[0,1]+isig[1,0])*xx*yy+isig[1,1]*yy*yy))
plt.figure()
plt.contour(xx,yy,p,cmap='hsv')
# plt.savefig('normdist2dc.eps')

# ax=Axes3D(plt.figure())
# ax.plot_wireframe(xx,yy,p);
# plt.savefig('normdist2dm.eps')
plt.show()
