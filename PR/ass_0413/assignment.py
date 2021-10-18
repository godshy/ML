from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

data = loadmat('./ML/PR/ass_0413/data2.mat')
x = data['x']  # data matrix
cov1 = data['cov1']  # covariance matrix 1
cov2 = data['cov2']  # covariance matrix 2
m1 = data['m1']  # mean vector 1
m2 = data['m2']  # mean vector 2


# contour

xx, yy = np.meshgrid(np.linspace(-5, 5), np.linspace(-5, 5))
xt1 = xx - m1[0, 0]
yt1 = yy - m1[0, 1]
xt2 = xx - m2[0, 0]
yt2 = yy - m2[0, 0]
icov1 = np.linalg.inv(cov1)
icov2 = np.linalg.inv(cov2)
p1 = 1./(2.*np.pi*np.sqrt(np.linalg.det(cov1))) * np.exp(-1./2.*(icov1[0, 0]*xt1*xt1+(icov1[0, 1]+icov1[1, 0])*xt1*yt1+icov1[1, 1]*yt1*yt1))
p2 = 1./(2.*np.pi*np.sqrt(np.linalg.det(cov2))) * np.exp(-1./2.*(icov2[0, 0]*xt2*xt2+(icov2[0, 1]+icov2[1, 0])*xt2*yt2+icov2[1, 1]*yt2*yt2))

#draw picture
plt.figure()
plt.axis('equal')
plt.scatter([x[:, 0]], [x[:, 1]])
plt.contour(xx,yy,p1,cmap='hsv')
plt.contour(xx,yy,p2,cmap='hsv')
plt.show()

#同時対角化の手順その１
cov1_eig, cov1_feature_vec = np.linalg.eig(cov1)  # BIG O and phi of cov1
tr1 = np.diag(cov1_eig**(-1./2.)).dot(cov1_feature_vec.T)
y1 = x.dot(tr1.T)  # whitening sigma1
Ident1 = tr1.dot(cov1).dot(cov1_feature_vec).dot(np.diag(cov1_eig**(-1./2.)))  # sigma1の変換
K1 = tr1.dot(cov2).dot(cov1_feature_vec).dot(np.diag(cov1_eig**(-1./2.)))  # sigma2の変換

cov2_eig, cov2_feature_vec = np.linalg.eig(cov2)  # BIG O and phi of cov1
tr2 = np.diag(cov2_eig**(-1./2.)).dot(cov2_feature_vec.T)
y2 = x.dot(tr2.T)  # whitening sigma1
Ident2 = tr2.dot(cov2).dot(cov2_feature_vec).dot(np.diag(cov2_eig**(-1./2.)))  # sigma1の変換
K2 = tr2.dot(cov1).dot(cov2_feature_vec).dot(np.diag(cov2_eig**(-1./2.)))  # sigma2の変換



#同時対角化の手順その２
lam1, psi1 = np.linalg.eig(K1)  #　K1の固有値行列と固有ベクトル行列lamda, psi を計算する
Z1 = y1.dot(psi1.T.T)


lam2, psi2 = np.linalg.eig(K2)  #　K2の固有値行列と固有ベクトル行列lamda, psi を計算する
Z2 = y2.dot(psi2.T.T)
# contour

m11 = np.mean(Z1, axis=0)
m22 = np.mean(Z2, axis=0)
m11 = np.matrix(m11)
m22 = np.matrix(m22)

xt11 = xx-m11[0, 0]
yt11 = yy-m11[0, 1]
xt22 = xx-m22[0, 0]
yt22 = yy-m22[0, 1]


lbd1 = psi1.T.dot(K1).dot(psi1)
lbd2 = psi2.T.dot(K2).dot(psi2)
iI1 = np.linalg.inv(Ident1)  # 対角化による得られた単位行列
iI2 = np.linalg.inv(Ident2)

ib1 = np.linalg.inv(lbd1)  # inverse of lambda of K1
ib2 = np.linalg.inv(lbd2) # K2

p11 = 1./(2.*np.pi*np.sqrt(np.linalg.det(lbd1))) * np.exp(-1./2.*(ib1[0,0]*xt11*xt11+(ib1[0,1]+ib1[1,0])*xt11*yt11+ib1[1,1]*yt11*yt11))
p22 = 1./(2.*np.pi*np.sqrt(np.linalg.det(lbd2))) * np.exp(-1./2.*(ib2[0,0]*xt22*xt22+(ib2[0,1]+ib2[1,0])*xt22*yt22+ib2[1,1]*yt22*yt22))


plt.figure()
plt.axis('equal')
plt.scatter([Z1[:, 0]], [Z1[:, 1]])
plt.contour(xx,yy,p11,cmap='hsv')
plt.contour(xx,yy,p22,cmap='hsv')
plt.show()
print("END")
