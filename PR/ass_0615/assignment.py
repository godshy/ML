import numpy as np
import matplotlib.pyplot as plt
import cvxopt

def lineardset():
    d = 2
    n = 100
    x = 2*np.random.rand(d, n)-np.array([1, 1])[:, np.newaxis]
    l = 2*((2*x[0, :]+x[1, :]) > 0.5) - 1
    return x, l, n

def slineardset():  # 有偏
    d=2
    n=500
    x=np.concatenate((np.random.randn(d,int(n/2))/4+np.array([0.5,0.5])[:,np.newaxis], \
                  np.random.randn(d,int(n/2))/4-np.array([0.5,0.5])[:,np.newaxis]), \
                 axis=1)
    l=2*(x[0,:]>0)-1
    return x, l, n

def quasilineardset():  # 拟线性
    d = 2
    n = 100
    x = 2*np.random.rand(d, n)-1
    l = 2*((2*x[0,:]+x[1, :]) > 0.5)-1
    flip = abs((2*x[0, :]+x[1, :])-0.5) < 0.2
    l[np.where(flip)] = -l[np.where(flip)]
    return x, l, n

def nonelineardset():  # 线性不可分
    d = 2
    n = 100
    x = 2*np.random.rand(d, n)-np.array([1, 1])[:, np.newaxis]
    l = 2*(((2*x[0, :]+x[1, :])>0.5) != ((x[0, :]-1.5*x[1, :]) > 0.5))-1
    return x, l, n


def soft_margin(x, l, n, C):
    h = x * l
    qpP = cvxopt.matrix(h.T.dot(h))
    qpq = cvxopt.matrix(-np.ones(n), (n, 1))
    qpG = cvxopt.matrix(np.vstack((np.eye(n) * -1, np.eye(n))))
    qph = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * C)))  # add a Constant C
    qpA = cvxopt.matrix(l.astype(float), (1, n))
    qpb = cvxopt.matrix(0.)
    cvxopt.solvers.options['abstol'] = 1e-5
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['show_progress'] = False
    res=cvxopt.solvers.qp(qpP, qpq, qpG, qph, qpA, qpb)
    alpha = np.reshape(np.array(res['x']), -1)
    w = np.sum(x*(np.ones(n)*(l*alpha)), axis=1)
    sv = alpha > 1e-5
    isv = np.where(sv)[-1]
    b = np.sum(w.T.dot(x[:, isv])-l[isv])/np.sum(sv)
    return w, b, sv

def hard_margin(x, l, n):
    h = x * l
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
    alpha = np.reshape(np.array(res['x']), -1)
    w = np.sum(x*(np.ones(n)*(l*alpha)), axis=1)
    sv = alpha > 1e-5
    isv = np.where(sv)[-1]
    b = np.sum(w.T.dot(x[:, isv])-l[isv])/np.sum(sv)
    return w, b, sv

def RBF_kernel(x, y, sigma):
    gamma = 1 / (2 * (sigma ** 2))
    return np.exp(-((x - y).T.dot(x-y)) * gamma)

def poly_kernel(x, y, beta):
    return (x * y) ** beta

def RBF_softmargin(x, l, n, C):
    phi = RBF_kernel(x, x, 0.5)
    h = x * l
    qpP = cvxopt.matrix(h.T.dot(h)* phi)  # xi を　phi(x)で書き換える
    qpq = cvxopt.matrix(-np.ones(n), (n, 1))
    qpG = cvxopt.matrix(np.vstack((np.eye(n) * -1, np.eye(n))))
    qph = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * C)))  #
    qpA = cvxopt.matrix(l.astype(float), (1, n))
    qpb = cvxopt.matrix(0.)
    cvxopt.solvers.options['abstol'] = 1e-5
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['show_progress'] = False
    res = cvxopt.solvers.qp(qpP, qpq, qpG, qph, qpA, qpb)
    alpha = np.reshape(np.array(res['x']), -1)
    w=np.sum(phi*(np.ones(n)*(l*alpha)), axis=1)   # 変えたw
    sv = alpha > 1e-5
    isv = np.where(sv)[-1]
    b = np.sum(w.T.dot(phi[:, isv])-l[isv])/np.sum(sv)  # 変えたb
    return w, b, sv
# b=np.sum(w.T.dot(x[:,isv])-l[isv])/np.sum(sv)

def polymonial_softmargin(x, l, n, C):
    h = x * l
    qpP = cvxopt.matrix(h.T.dot(h) * poly_kernel(x, x, 3))
    qpq = cvxopt.matrix(-np.ones(n), (n, 1))
    qpG = cvxopt.matrix(np.vstack((np.eye(n) * -1, np.eye(n))))  # G
    qph = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * C)))  # h
    qpA = cvxopt.matrix(l.astype(float), (1, n))
    qpb = cvxopt.matrix(0.)
    cvxopt.solvers.options['abstol'] = 1e-5
    cvxopt.solvers.options['reltol'] = 1e-10
    cvxopt.solvers.options['show_progress'] = False
    res=cvxopt.solvers.qp(qpP, qpq, qpG, qph, qpA, qpb)
    alpha = np.reshape(np.array(res['x']), -1)
    w = np.sum(np.ones(n)*(l*alpha).dot(poly_kernel(x, x, 3)))
    sv = alpha > 1e-5
    isv = np.where(sv)[-1]
    b = np.sum(w.T.dot(poly_kernel(x, x, 3)[:, isv])-l[isv])/np.sum(sv)
    return w, b, sv


def plot(x, w, b, sv, l):
    plt.figure()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.plot(x[0, np.where((l > 0) & sv)], x[1, np.where((l>0) & sv)], 'bo')
    plt.plot(x[0, np.where((l > 0) & ~sv)], x[1, np.where((l>0) & ~sv)], 'bx')
    plt.plot(x[0, np.where((l < 0) & sv)], x[1, np.where((l<0) & sv)], 'ro')
    plt.plot(x[0, np.where((l < 0) & ~sv)], x[1, np.where((l<0) & ~sv)], 'rx')
    if abs(w[0]) > abs(w[1]):
        plt.plot([-1, 1], [(b+1+w[0])/w[1], (b+1-w[0])/w[1]])
        plt.plot([-1, 1], [(b-1+w[0])/w[1], (b-1-w[0])/w[1]])
    else:
        plt.plot([(b+1+w[1])/w[0], (b+1-w[1])/w[0]], [-1, 1])
        plt.plot([(b-1+w[1])/w[0], (b-1-w[1])/w[0]], [-1, 1])
    plt.show()
def f(x, w, b, c):
        # given x, return y such that [x,y] in on the line
        # w.x + b = c
    return (-w[0] * x - b + c) / w[1]

def RBF_plot(x, w, b, sv, l):
    plt.figure()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.plot(x[0, np.where((l > 0) & sv)], x[1, np.where((l>0) & sv)], 'bo')
    plt.plot(x[0, np.where((l > 0) & ~sv)], x[1, np.where((l>0) & ~sv)], 'bx')
    plt.plot(x[0, np.where((l < 0) & sv)], x[1, np.where((l<0) & sv)], 'ro')
    plt.plot(x[0, np.where((l < 0) & ~sv)], x[1, np.where((l<0) & ~sv)], 'rx')

    a1 = f(-1, w, b, 0)
    b1 = f(1, w, b, 0)
    print(a1, b1)
    plt.plot([-1, 1], [a1, b1])
    a2 = f(-1, w, b, 1)
    b2 = f(1, w, b, 1)
    plt.plot([-1, 1], [a2, b2])
    a3 = f(-1, w, b, -1)
    b3 = f(1, w, b, -1)
    plt.plot([-1, 1], [a3, b3])
    plt.show()

'''  
    plt.plot(np.dot(x[0, np.where((l > 0) & sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]), np.dot(x[1, np.where((l>0) & sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]), 'bo')
    plt.plot(np.dot(x[0, np.where((l > 0) & ~sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]), np.dot(x[1, np.where((l>0) & ~sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)] ), 'bx')
    plt.plot(np.dot(x[0, np.where((l < 0) & sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]), np.dot(x[1, np.where((l<0) & sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]),'ro')
    plt.plot(np.dot(x[0, np.where((l < 0) & ~sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]), np.dot(x[1, np.where((l<0) & ~sv)].T, RBF_kernel(x, x, sig)[0, np.where((l > 0) & sv)]), 'rx')
'''

x_line, l_line, n_l = lineardset()  # 2x100 x, 100x1 l
x_sline, l_sline, n_s = slineardset()  # 2x500 x, 500x1 l
x_qline, l_qline, n_q = quasilineardset()  # 2x100 x, 100x1 l
x_nline, l_nline, n_n = nonelineardset()  # 2x100 x, 100x1 l

w_n, b_n, sv_n = RBF_softmargin(x_qline, l_qline, n_n, 50)
RBF_plot(x_qline, w_n, b_n, sv_n, l_qline)



C = np.array([20, 50, 100])
'''
w_n_sig, b_n_sig, sv_n_sig = polymonial_softmargin(x_nline, l_nline, n_n, 3)
RBF_plot(x_nline, w_n_sig, b_n_sig, sv_n_sig, l_nline)

w_h, b_h, sv_h = hard_margin(x_line, l_line, n_l)
plot(x_line, w_h, b_h, sv_h, l_line)
for i in range(3):
    w_s, b_s, sv_s = soft_margin(x_line, l_line, n_l, C[i])
    plot(x_line, w_s, b_s, sv_s, l_line)



w_hs, b_hs, sv_hs = hard_margin(x_sline, l_sline, n_s)
plot(x_sline, w_hs, b_hs, sv_hs, l_sline)
w_ss, b_ss, sv_ss = soft_margin(x_sline, l_sline, n_s, 50)
plot(x_sline, w_ss, b_ss, sv_ss, l_sline)

w_hq, b_hq, sv_hq = hard_margin(x_qline, l_qline, n_q)
plot(x_qline, w_hq, b_hq, sv_hq, l_qline)
w_sq, b_sq, sv_sq = soft_margin(x_qline, l_qline, n_q, 50)
plot(x_qline, w_sq, b_sq, sv_sq, l_qline)
'''

