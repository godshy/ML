import numpy as np
import matplotlib.pyplot as plt
import cvxopt


d=2
n=100
x=2*np.random.rand(d,n)-np.array([1,1])[:,np.newaxis]
l=2*((2*x[0,:]+x[1,:])>0.5)-1


h=x*l

qpP = cvxopt.matrix(h.T.dot(h))

qpq = cvxopt.matrix(-np.ones(n), (n, 1))
qpG = cvxopt.matrix(-np.eye(n))
qph = cvxopt.matrix(np.zeros(n), (n, 1))
qpA = cvxopt.matrix(l.astype(float), (1, n))
qpb = cvxopt.matrix(0.)
cvxopt.solvers.options['abstol'] = 1e-5
cvxopt.solvers.options['reltol'] = 1e-10
cvxopt.solvers.options['show_progress'] = False
res=cvxopt.solvers.qp(qpP, qpq, qpG, qph, qpA, qpb)
alpha = np.reshape(np.array(res['x']),-1)
temp = np.ones(n)*(l*alpha)

w=np.sum(x*(np.ones(n)*(l*alpha)), axis=1)
sv=alpha>1e-5
isv=np.where(sv)[-1]
b=np.sum(w.T.dot(x[:,isv])-l[isv])/np.sum(sv)


plt.figure()
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.plot(x[0,np.where((l>0) & sv)], x[1,np.where((l>0) & sv)],'bo')
plt.plot(x[0,np.where((l>0) & ~sv)], x[1,np.where((l>0) & ~sv)],'bx')
plt.plot(x[0,np.where((l<0) & sv)], x[1,np.where((l<0) & sv)],'ro')
plt.plot(x[0,np.where((l<0) & ~sv)], x[1,np.where((l<0) & ~sv)],'rx')
if abs(w[0])>abs(w[1]):
    plt.plot([-1, 1],[(b+1+w[0])/w[1], (b+1-w[0])/w[1]])
    plt.plot([-1, 1],[(b-1+w[0])/w[1], (b-1-w[0])/w[1]])
    print((b+1+w[0])/w[1], (b+1-w[0])/w[1])
else:
    plt.plot([(b+1+w[1])/w[0], (b+1-w[1])/w[0]], [-1, 1])
    plt.plot([(b-1+w[1])/w[0], (b-1-w[1])/w[0]], [-1, 1])
plt.show()