import numpy as np
import matplotlib.pyplot as plt


def gausscontour(c, m, xx, yy):
    xt = xx - m[0, 0]
    yt = yy-m[1, 0]
    ic = np.linalg.inv(c)
    p = 1./(2.*np.pi*np.sqrt(np.linalg.det(c))) * \
        np.exp(-1./2.*(ic[0, 0]*xt*xt+(ic[0, 1]+ic[1, 0])*xt*yt+ic[1, 1]*yt*yt))
    return p


d = 2
n = 1000
# m1 = np.array([0., 2.])[:, np.newaxis]
# m2 = np.array([3., 0.])[:, np.newaxis]

p1 = 0.3
p2 = 1-p1
cov1 = np.eye(2)  # 単位行列
cov2 = cov1
x1 = np.matrix(np.random.randn(n, d))
x2 = np.matrix(np.random.randn(n, d))
m1 = np.mean(x1, axis=0)
m2 = np.mean(x2, axis=0)

cov1 = (x1-np.ones([n, 1]).dot(m1)).T.dot(x1-np.ones([n, 1]).dot(m1))/n
cov2 = (x2-np.ones([n, 1]).dot(m2)).T.dot(x2-np.ones([n, 1]).dot(m2))/n

ed1, ev1 =np.linalg.eig(cov1)
tr1 = np.diag(ed1 ** (-1./2.)).dot(ev1.T)
x1_aft = x1.dot(tr1.T)
cov1_aft =tr1.dot(cov1).dot(tr1.T)
m1_aft = np.mean(x1, axis=0)

ed2, ev2 =np.linalg.eig(cov2)
tr2 = np.diag(ed2 ** (-1./2.)).dot(ev2.T)
x2_aft = x2.dot(tr2.T)
cov2_aft =tr2.dot(cov2).dot(tr2.T)
m2_aft = np.mean(x2, axis=0)

x1 = x1.T
x2 = x2.T
m1 = m1.T
m2 = m2.T
cov1 = cov1.T
cov2 = cov2.T


x1_aft = x1_aft.T
x2_aft = x2_aft.T
m1_aft = m1_aft.T
m2_aft = m2_aft.T
cov1_aft = cov1_aft.T
cov2_aft = cov2_aft.T
cov2_aft = cov1_aft
# x1 = np.random.randn(d, n)+m1.dot(np.ones([1, n]))
# x2 = np.random.randn(d, n)+m2.dot(np.ones([1, n]))  # 線形変換 case 2, 3


# 識別境界線
w = m1_aft -m2_aft
x0 = 1./2.*(m1_aft + m2_aft)-1./np.linalg.norm(m1_aft-m2_aft)**2.*np.log(p1/p2)*(m1_aft-m2_aft)
l1 = (w.T.dot(x1_aft-x0)>0)[-1]  # 正しい識別されたのか
l2 = (w.T.dot(x2_aft-x0)>0)[-1]

[xx,yy]=np.meshgrid(np.linspace(-2,5),np.linspace(-2,5))
plt.figure()
plt.axis('equal')
p1 = gausscontour(cov1_aft, m1_aft, xx, yy)
plt.contour(xx,yy,p1,cmap='hsv')
p2=gausscontour(cov2_aft,m2_aft,xx,yy)
plt.contour(xx,yy,p2,cmap='hsv')
# correct x1
plt.plot(x1_aft[0,np.where(l1)],x1_aft[1,np.where(l1)],'bo')
# wrong x1
plt.plot(x1_aft[0,np.where(~l1)],x1_aft[1,np.where(~l1)],'ro')
# correct x2
plt.plot(x2_aft[0,np.where(1-l2)],x2_aft[1,np.where(1-l2)],'r^')
# wrong x2
plt.plot(x2_aft[0,np.where(l2)],x2_aft[1,np.where(l2)],'b^')
xxyy=np.c_[np.reshape(xx,-1),np.reshape(yy,-1)].T
pp=w.T.dot(xxyy-x0*np.ones([1,xxyy.shape[1]]))
pp=np.reshape(pp,xx.shape)
cs=plt.contour(xx,yy,pp,cmap='hsv')
plt.clabel(cs)
#plt.savefig('cov_diag.eps')
plt.show()
