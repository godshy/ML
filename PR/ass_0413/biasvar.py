import numpy as np
import matplotlib.pyplot as plt

m=1000
nn=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
v=np.zeros(len(nn)); e=np.zeros(len(nn))
vn=np.zeros(len(nn)); en=np.zeros(len(nn))
vv=np.zeros(m); vvn=np.zeros(m)
ideal=1./3.-0.5+0.5**2.
for i in range(len(nn)):
  n=nn[i]
  for j in range(m):
    r=np.random.rand(int(n))
    r=r-np.mean(r)
    r=r*r
    vv[j]=np.sum(r)/n;
    vvn[j]=np.sum(r)/(n-1);
  v[i]=np.mean(vv)
  e[i]=np.std(vv)
  vn[i]=np.mean(vvn)
  en[i]=np.std(vvn)
plt.clf()
if 0:
  plt.errorbar(nn,v,yerr=e,label='biased')
  plt.errorbar(nn,vn,yerr=en,label='unbiased')
else:
  plt.plot(nn,v,label='biased')
  plt.plot(nn,vn,'--',label='unbiased')
plt.semilogx(nn,ideal*np.ones(len(v)),'r-',label='ideal')
plt.legend(loc="lower right")
plt.xlabel('sample size')
plt.ylabel('estimated variance')
# plt.savefig('biasvar.eps')
plt.show()
