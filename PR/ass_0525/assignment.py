
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def qclass(m1,cov1,m2,cov2,p1,p2,x1,x2):
  icov1=np.linalg.inv(cov1)
  icov2=np.linalg.inv(cov2)
  W1=-1./2.*icov1
  W2=-1./2.*icov2
  w1=icov1@m1
  w2=icov2@m2
  w10=m1.T@W1@m1 - 1./2.*np.log(np.linalg.det(cov1))+np.log(p1)
  w20=m2.T@W2@m2 - 1./2.*np.log(np.linalg.det(cov2))+np.log(p2)
  g11=np.diag(x1@W1@x1.T)+w1.T@x1.T+w10
  g12=np.diag(x2@W1@x2.T)+w1.T@x2.T+w10
  g21=np.diag(x1@W2@x1.T)+w2.T@x1.T+w20
  g22=np.diag(x2@W2@x2.T)+w2.T@x2.T+w20
  l1=((g11-g21)>0)
  l2=((g12-g22)>0)
  err=(np.count_nonzero(~l1)+np.count_nonzero(l2))/(np.size(l1)+np.size(l2))
  return err




def bootstrap(x1, x2, p1, p2,jdim, jn, js, exp, dim, Rerrs, Berrs, Berrs_hat):

  nboot = np.array([5, 10])    # 　仮想セットを　B　回生成する。
  Berrs_f = []
  for i in range(nboot.size):
    b_star = []
    for j in range(nboot[i]):
#      s_stars_x1 = np.zeros([j])  #hape:1 x nboot with 1 x len(x1 or x2)
#      s_stars_x2 = np.zeros([j])
      s_star_x1 = np.zeros([np.size(x1[:, 0]), dim])
      s_star_x2 = np.zeros([np.size(x2[:, 0]), dim])
      for kk in range(dim):
        for k in range(np.size(x1[:, 0])):
          s_star_x1[k, kk] = x1[np.random.randint(0, np.size(x1[:, 0])), kk]    # x1 x2 それぞれを対し |S|回の復元抽出を行う。
      for ll in range(dim):
        for l in range(np.size(x2[:,0])):
          s_star_x2[l, ll] = x2[np.random.randint(0, np.size(x2[:, 0])), ll]

#      s_stars_x1[j] = s_star_x1
#      s_stars_x2[j] = s_star_x2

      m_x1_stars = np.mean(s_star_x1, axis=0)
      m_x2_stars = np.mean(s_star_x2, axis=0)
      cov1_x1_stars = (s_star_x1-np.ones([np.size(x1[:, 0]),1])*m_x1_stars).T@(s_star_x1-np.ones([np.size(x1[:, 0]),1])*m_x1_stars )/(np.size(x1[:, 0])-1)
      cov2_x2_stars = (s_star_x2-np.ones([np.size(x2[:, 0]),1])*m_x2_stars).T@(s_star_x2-np.ones([np.size(x2[:, 0]),1])*m_x2_stars )/(np.size(x2[:, 0])-1)

      Berrs[jdim,jn,js,exp] = qclass(m_x1_stars,cov1_x1_stars,m_x2_stars,cov2_x2_stars,p1,p2,x1,x2)  # S*を学習セット、Sを評価セット、epsilon*をエル
      Berrs_hat[jdim,jn,js,exp] = qclass(m_x1_stars,cov1_x1_stars,m_x2_stars,cov2_x2_stars,p1,p2,s_star_x1, s_star_x2)
      c = Berrs - Berrs_hat  # b* を得る
      b_star.append(c)
     # b_star[] = numpy.concatenate([b_star, c], axis=3)

    b_star = np.array(b_star)
    Berrs_f.append((Rerrs + np.mean(b_star, axis=0)))  # epsilon　を計算する
  return Rerrs, Berrs_f




#   rerr=compute error by R method with x1 and x2
#   bbias=np.zeros(nboot)
#   for b in range(nboot):
#     generate x1b and x2b by sampling with replacement
#     erb1=train with x1b and x2b（s*）, and test with x1 and x2 (s)
#     erb2=train with x1b and x2b, and test with x1b and x3b (s*)
#     bbias[b]=erb1-erb2
#   return rerr, rerr+np.mean(bbias)

# nboot=5, 10, or 20, ...

if 1:
  dims=np.array([10, 20, 25]) # dimension
  ns=np.array([300, 500]) # num samples
  fts=np.array([0.2, 0.5, 0.8]) # fraction of training for H
  defdim=2; defn=1; defft=1
else:
  dims=np.array([10, 20]) # dimension
  ns=np.array([300, 1000]) # num samples
  fts=np.array([0.2, 0.5, 0.8]) # fraction of training for H
  defdim=2; defn=1; defft=1
ss=np.array([0.4, 0.6, 0.8, 1.0, 1.2]) # offsets
defs=3
nexp=10 # number of experiments
p1=0.5
p2=1-p1
Rerrs=np.zeros([len(dims),len(ns),len(ss),nexp])
Herrs=np.zeros([len(dims),len(ns),len(fts),len(ss),nexp])
berrs=np.zeros(len(ss))
Berrs = np.zeros([len(dims),len(ns),len(ss),nexp])
Berrs_hat = np.zeros([len(dims),len(ns),len(ss),nexp])
Berrs_f = np.zeros([4, 1, len(dims),len(ns),len(ss),nexp])

for (jdim,dim) in enumerate(dims):
  print('dim=%d' % dim)
  for (js,s) in enumerate(ss):
    off=np.zeros(dim)
    off[0]=s
    berrs[js]=1./2.*scipy.special.erfc(s/np.sqrt(2.))
    for exp in range(nexp):
      for (jn,n) in enumerate(ns):
        for (jft,ft) in enumerate(fts):
          ntr=int(n*ft)
          nte=n-ntr
          x1tr=np.random.randn(ntr,dim)+off
          x2tr=np.random.randn(ntr,dim)-off
          x1te=np.random.randn(nte,dim)+off
          x2te=np.random.randn(nte,dim)-off
          m1tr=np.mean(x1tr,axis=0)
          m2tr=np.mean(x2tr,axis=0)
          cov1tr=(x1tr-np.ones([ntr,1])*m1tr).T@(x1tr-np.ones([ntr,1])*m1tr)/(ntr-1)
          cov2tr=(x2tr-np.ones([ntr,1])*m2tr).T@(x2tr-np.ones([ntr,1])*m2tr)/(ntr-1)
          Herrs[jdim,jn,jft,js,exp]=qclass(m1tr,cov1tr,m2tr,cov2tr,p1,p2,x1te,x2te)
        x1=np.concatenate([x1tr,x1te],axis=0)
        x2=np.concatenate([x2tr,x2te],axis=0)
        m1=np.mean(x1,axis=0)
        m2=np.mean(x2,axis=0)
        cov1=(x1-np.ones([n,1])*m1).T@(x1-np.ones([n,1])*m1)/(n-1)
        cov2=(x2-np.ones([n,1])*m2).T@(x2-np.ones([n,1])*m2)/(n-1)
        Rerrs[jdim,jn,js,exp]=qclass(m1,cov1,m2,cov2,p1,p2,x1,x2)
        (Rerr, Berrs_f)=bootstrap(x1, x2, p1, p2, jdim, jn,
                                  js, exp, dim, Rerrs, Berrs, Berrs_hat)

# def bootstrap(x1, x2, p1, p2,jdim, jn, js, exp, dim, Rerrs, Berrs, Berrs_hat):
Berrs_f = np.array(Berrs_f)
print(Berrs_f.shape)
Rmean=np.mean(Rerrs,axis=3)
Rstd=np.std(Rerrs,axis=3)
Hmean=np.mean(Herrs,axis=4)
Hstd=np.std(Herrs,axis=4)
Bmean = np.mean(Berrs_f, axis=4)  #
Bstd = np.std(Berrs_f, axis=4)  #
nboot = np.array([5, 10]) #

plt.figure()
# plt.errorbar(ss,Rmean[defdim,defn,:],yerr=Rstd[defdim,defn,:],label='R')
plt.errorbar(ss,Rmean[defdim,defn,:],yerr=Rstd[defdim,defn,:],label='R')
for (jft,ft) in enumerate(fts):
  plt.errorbar(ss,Hmean[defdim,defn,jft,:],
               yerr=Hstd[defdim,defn,jft,:],label='H(%g)' % ft)
for (i, j) in enumerate(nboot):
  plt.errorbar(ss, Bmean[i,defdim,defn,:],
               yerr=Bstd[i,  defdim, defn,:],label='nboot(%g)' % j)
plt.plot(ss,berrs[:],label='Bayes')
plt.legend()
plt.xlabel('off')
plt.ylabel('err')
plt.title('dim=%d n=%d' % (dims[defdim],ns[defn]))

plt.figure()
plt.errorbar(dims,Rmean[:,defn,defs],yerr=Rstd[:,defn,defs],label='R')
for (jft,ft) in enumerate(fts):
  plt.errorbar(dims,Hmean[:,defn,jft,defs],yerr=Hstd[:,defn,jft,defs],label='H(%g)' % ft)
for (i, j) in enumerate(nboot):
  plt.errorbar(dims, Bmean[i,  :, defn, defs], yerr=Bstd[i,  :, defn, defs],label='nboot(%g)' % j)
plt.plot(dims,berrs[defs]*np.ones(len(dims)),label='Bayes')
plt.legend()
plt.xlabel('dim')
plt.ylabel('err')
plt.title('n=%d off=%g' % (ns[defn],ss[defs]))


plt.figure()
plt.errorbar(ns,Rmean[defdim,:,defs],yerr=Rstd[defdim,:,defs],label='R')
for (jft,ft) in enumerate(fts):
  plt.errorbar(ns,Hmean[defdim,:,jft,defs],yerr=Hstd[defdim,:,jft,defs],label='H(%g)' % ft)
for (i, j) in enumerate(nboot):
  plt.errorbar(ns, Bmean[i,  defdim, :, defs], yerr=Bstd[i,  defdim, :, defs],label='nboot(%g)' % j)
plt.plot(ns,berrs[defs]*np.ones(len(ns)),label='Bayes')
plt.legend()
plt.xlabel('n')
plt.ylabel('err')
plt.xscale('log')
plt.title('dim=%d off=%g' % (dims[defdim],ss[defs]))


plt.show()
