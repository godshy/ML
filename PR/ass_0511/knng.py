import numpy as np
import matplotlib.pyplot as plt

# nl: list of n (num. of points)
nl=np.array([1, 16, 256, 65536])
x=np.linspace(-3., 3., num=500)
plt.figure(figsize=(8, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.3)
for i in range(len(nl)):
    n=nl[i]
    k=np.sqrt(n)
    # s: list of n points yielding Gaussian
    s=np.random.randn(n)
    p=np.zeros(len(x))
    for j in range(len(x)):
        # r: sorted list by the distance to x[j]
        r = sorted(abs(s-x[j]))
	# r[int(k)-1]: k-th distance
        p[j]=k/(n*2.*r[int(k)-1])
    ax=plt.subplot(2,2,i+1,title="n="+str(nl[i]))
    print(x.shape,  p.shape)
    plt.plot(x,p)
    if k==1:
        ax.set_ylim([0, 10])
plt.show()
