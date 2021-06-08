import numpy as np
import matplotlib.pyplot as plt

def gausscontour(c,m,xx,yy):
    xt=xx-m[0,0]
    yt=yy-m[1,0]
    ic=np.linalg.inv(c)
    p=1./(2.*np.pi*np.sqrt(np.linalg.det(c))) * \
        np.exp(-1./2.*(ic[0,0]*xt*xt+(ic[0,1]+ic[1,0])*xt*yt+ic[1,1]*yt*yt))
    return p


d = 2; n = 1000
m1 = np.array([0., 2.])[:, np.newaxis]
m2 = np.array([3., 0.])[:, np.newaxis]
p1 = 0.3
p2 = 1-p1
cov1 = np.eye(2)  # 単位行列
cov2 = cov1
x1 = np.random.randn(d, n)+m1.dot(np.ones([1, n]))
x2 = np.random.randn(d, n)+m2.dot(np.ones([1, n]))  # 線形変換 case 2, 3


# 識別境界線
w = m1-m2
x0 = 1./2.*(m1+m2)-1./np.linalg.norm(m1-m2)**2.*np.log(p1/p2)*(m1-m2)
l1 = (w.T.dot(x1-x0)>0)[-1]  # 正しい識別されたのか
l2 = (w.T.dot(x2-x0)>0)[-1]

[xx,yy]=np.meshgrid(np.linspace(-2,5),np.linspace(-2,5))
plt.figure()
plt.axis('equal')
p1=gausscontour(cov1,m1,xx,yy)
plt.contour(xx,yy,p1,cmap='hsv')
p2=gausscontour(cov2,m2,xx,yy)
plt.contour(xx,yy,p2,cmap='hsv')
# correct x1
plt.plot(x1[0,np.where(l1)],x1[1,np.where(l1)],'bo')
# wrong x1
plt.plot(x1[0,np.where(~l1)],x1[1,np.where(~l1)],'ro')
# correct x2
plt.plot(x2[0,np.where(1-l2)],x2[1,np.where(1-l2)],'r^')
# wrong x2
plt.plot(x2[0,np.where(l2)],x2[1,np.where(l2)],'b^')
xxyy=np.c_[np.reshape(xx,-1),np.reshape(yy,-1)].T
pp=w.T.dot(xxyy-x0*np.ones([1,xxyy.shape[1]]))
pp=np.reshape(pp,xx.shape)
cs=plt.contour(xx,yy,pp,cmap='hsv')
plt.clabel(cs)
#plt.savefig('cov_diag.eps')
plt.show()
