import numpy as np


def assignUserPostions(Nuser, type, Wx, Wy, Nx, Ny):
    # uniform distribution
    if type == 'uniform':
        usrpositions = np.zeros(Nuser, 2)
        # np.random.seed(0)
        p1 = np.random.rand(Nuser).reshape(1, -1)

        usrpositions[:, :1] = Wx * np.transpose(p1)
        usrpositions[:, :2] = Wy * np.transpose(p1)
        return usrpositions

    elif type == 'all':  # not uniform distribution
        width1 = np.floor(Wx/Nx)
        width2 = np.floor(Wy/Ny)

        usrpositions = np.zeros(Nx * Ny, 2)
        for i1 in range(1, Nx):
            for i2 in range(1, Ny):
                usrpositions[i1 + (i2 - 1) * Nx:, :1] = (i1 - 0.5) * width1
                usrpositions[i1 + (i2 - 1) * Nx:, :2] = (i2 - 0.5) * width2
        return usrpositions
    else:
        return '確率分布を正しく指定してください'


def assignAPPositions(type, Nap, N1, N2, Wx, Wy):
    APpositions = np.zeros(Nap, 2)
    if type == 'uniform':
        p1 = np.random.rand(Nap).reshape(1, -1)
        APpositions[:, :1] = Wx * np.transpose(p1)
        APpositions[:, :2] = Wy * np.transpose(p1)
        return APpositions
    elif type == 'grid':
        width1 = Wx/N1
        width2 = Wy/N2
        for i1 in range(1, N1):
            for i2 in range(1, N2):
                APpositions[i1 + (i2 - 1) * N1, :1] = (i1 - 0.5) * width1
                APpositions[i1 + (i2 - 1) * N1, :2] = (i2 - 0.5) * width2

        return APpositions
    else:
        return '確率分布を正しく指定してください'


# path loss
def setpathloss(apPosition, usrPosition, freq):
    d0 = 10 * 1e-3
    d1 = 50 * 1e-3
    hap = 30  # ap height
    hu = 1.65  # user height
    dist = np.linalg.norm(usrPosition - apPosition, ord=2) * 1e-3

    if (dist > d1):
        ploss = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(hap) - (1.1 * np.log10(freq) - 0.7) * hu + (1.56 * np.log10(freq) - 0.8) + 35 * np.log10(dist)
    elif(dist <= d1 and d1 > 0):
        ploss = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(hap) - (1.1 * np.log10(freq) - 0.7) * hu + (1.56 * np.log10(freq) - 0.8)+ 15 * np.log10(d1) + 20 * np.log10(dist)
    else:
        ploss = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(hap) - (1.1 * np.log10(freq) - 0.7) * hu + (1.56 * np.log10(freq) - 0.8) + 15 * np.log10(d1) + 20 * np.log10(d0);
    return ploss

def makechannel(type, PathLoss, apPosition, userPosition, sigmaS, deltaS, ddecorr, pathlossDB, Nap, Nuser):
    apusr = {'Nap': PathLoss[:, 0],'Nusr': PathLoss[:,1]}
    if type == 'correlated':
        CorAP = np.eye(Nap, Nap)
        for i1 in range(1, Nap):
            for i2 in range(i1+1, Nap) :
                CorAP[i1, i2] = 2.0^(-1.0 * np.linalg.norm(apPosition[i1,:] - apPosition[i2, :], ord=2)/ddecorr)
                CorAP[i2, i1] = CorAP[i1, i2]
        CorUser = np.eye(Nuser, Nuser)
        for i1 in range(1, Nuser):
            for i2 in range(i1+1 , Nuser):
                CorUser[i1, i2] = 2.0^(1.0 * np.linalg.norm(userPosition[i1, :] - userPosition[i2, :], ord=2)/ddecorr)
                CorUser[i2, i1] = CorAP[i1, i2]

    elif type == 'uncorrelated':
        CorAP = np.eye(Nap, Nap)
        CorUser = np.eye(Nuser, Nuser)

    Betaout = np.zeros(Nap, Nuser)

    if type == 'pathloss':
        zsh1 = np.zeros(Nap, 1)
        zsh2 = np.zeros(Nuser, 1)
    else:
        tmpR1 = np.linalg.cholesky(CorAP)
        tmpR2 = np.linalg.cholesky(CorUser)
        p1 =  np.random.rand(Nap).reshape(1, -1)
        zsh1 = tmpR1 * p1
        p2 = np.random.rand(Nuser).reshape(1, -1)
        zsh2 = tmpR2 * p2

    for m in range(1, Nap):
        for k in range(1, Nuser):
            if np.linalg.norm(userPosition[k, :] - apPosition[m, :], ord=2) < 50:
                Betaout[m, k] = 10^(-1.0 * pathlossDB[m, k]/10)
            else:
                zsh = np.sqrt(deltaS) * zsh1[m, 1] + np.sqrt(1 - deltaS) * zsh2[k, 1]
                Betaout[m, k] = 10^(-1.0 * pathlossDB[m, k]/10) * 10^(sigmaS * zsh/10)

    # return



def generatePilot(tau):

    for i1 in range(1, tau):
        for i2 in range(1, tau):
            Q = np.zeros(i1, i2)
            Q[i1, i2] = 1/np.sqrt(tau) * np.exp(2 * np.pi * np.random.rand().reshape(1, -1))  # don't know if it works

    return np.linalg.qr(Q)

def pilotAssignment(type, Q, Nusr, tau):

    new = np.array([tau, Nusr])
    if type == 'roundrobin':
        P = np.zeros(tau, Nusr)
        for k in range(1, Nusr):
            pilotid = np.mod(k, tau) + 1
            P[:, :k] = Q[:, pilotid]
            return P

    else:
        return "ERROR"


def makeMMSE(tau, rhop, Beta, P, Nap, Nuser):   # 有问题

    Cout = np.zeros(Nap, Nuser)
    for m in range(1, Nap):
        for k in range(1, Nuser):
            tmpc = 0
            for i in range(1, Nuser):
                tmpc = tmpc + Beta[m, i] * np.abs(P[:, k] * P[:, i]) ** 2

            Cout[m, k] = np.sqrt(tau * rhop) * Beta[m, k]/(tau * rhop * tmpc + 1)

    return Cout

def calcGamma(Beta, C, tau, rhop):
    [m, k] = np.size(Beta)
    gammaout = np.zeros(m, k)
    for m in range(1, m):
        for k in range(1, k):
            gammaout[m, k] = np.sqrt(tau * rhop) * Beta[m, k] *C[m, k]

    return  gammaout

def mat2vec(x, M, k):
    outvec = np.zeros(M * k, 1)
    for m in range(1, M):
        for k in range(1, k):
            outvec[k + (m-1) * k, 1] = x[m, k]

def vec2mat(x, M, K):
    outmat = np.zeros(M, K);
    for m in range(1, M):
        for k in range(1, K):
            outmat[M, K] = x[k + (m - 1) * K, 1]

    return outmat





def BisecObjFunc(x, beta, gammad, p, rhod):
    [m, k] = np.size(beta)
    [L, ] = np.size(x)

    avec = x[1:(m * k), 1]
    bvec = x[m * k + 1 : m * k + k * k, 1]
    cvec = x[ m * k + k * k + 1: L, 1]
    amat = vec2mat(avec, m, k)
    bmat = vec2mat(bvec, m, k)

    snr = np.zeros(k, 1)
    for k in range(1, k):
        tmp1 = 0
        for m in range(1, m):
            tmp1 += gammad[m, k] * amat[m, k]
        tmp1 = tmp1 **2
        tmp2 = 0

        for i in range(1, k):
            if (1 != k):
                tmp2 += np.abs(p[:, i] * p[:, k]) ** 2 * bmat[i, k] ** 2

        tmp3 = 0

        for m in range(1, m):
            tmp3 += beta[m, k] * cvec[m, 1] ** 2


    snr[k, 1] = tmp1/(tmp2 + tmp3 + 1.0/rhod)

    return np.min(snr)





def BisecNonlcon(x, gammad, M, K):
    [m, k] = np.size(gammad)
    L = M * K + K ** 2 + M;

    avec = x[1:M*K, 1]
    bvec = x[M * K + 1: M * K + K * K, 1]
    cvec = x[M * K + K * K + 1: L, 1]
    amat = vec2mat(avec, M, K)
    bmat = vec2mat(bvec, K, K)

    cineq = np.zeros(M, 1)
    for m in range(1, m):
        for i in range(1, k):
            cineq[m, 1] += gammad[m, i] * amat[m, i] ** 2 - cvec[m, 1] ** 2
    # ceq = []


def makeTXPcontrolD( type, pctype, Beta, C, tau, rhop, rhod, P):

    [m, k] = np.size(Beta)
    etadout = np.zeros(m, k)

    if type == 'without':
        gammad = calcGamma(Beta, C, tau, rhop)

        for k in range(1, k):
            for m in range(1, m):
                tempeta = 0
                for i in range(1, k):
                    tempeta += gammad[m, i]
                etadout[m, k] = 1/tempeta

    elif type == 'random':
        gammad = calcGamma(Beta, C, tau, rhop)
        etad0 = np.random.random(m, k)
        for m in range(m, k):
            tmpsum = 0
            for k in range(1, k):
                tmpsum += etad0[m, k] * gammad[m, k]
            etad0[m, :] = etad0[m, :]/tmpsum
        etadout = etad0


    elif type == 'optimum':
        gammad = calcGamma(Beta, C, tau, rhop)

        etad0 = np.zeros(m, k)
        for k in range(1, k):
            for m in range(1, m):
                tempeta = 0
                for i in range(1, k):
                    tempeta += gammad[m, k]  #  tmpeta = tmpeta + gammad(m, i);
                etad0[m, k] = 1/tempeta


        tmpmat = np.zeros(m, k)
        for m in range(1, m):
            for k in range(1, k):
                tmpmat[m, k] = np.sqrt(etad0[m, k])

        avec = mat2vec(tmpmat, m, k)
        bvec = np.ones(k * k, 1)
        cvec = 1 * np.ones(m, 1)

        x0 = np.vstack(np.vstack(avec, bvec), cvec)

        #objfunc = x -1.0 * BisecObjFunc(x, Beta, gammad, P, rhod)
