import numpy as np
import matplotlib.pyplot as plt

p=0.7
n=np.array([20, 50, 100])
m=100000
s=('-','o-','x-')
leg=('n=20','n=50','n=100')
plt.figure()
for i in range(len(n)):
  c=np.zeros(n[i]+1)
  k=np.sum(np.random.rand(n[i],m)<=p, axis=0)
  for j in range(m):
    c[k[j]] += 1
  c=c/m
  c=c/np.max(c)
  plt.plot(np.arange(n[i]+1)/n[i], c, s[i], label=leg[i])
plt.legend(loc='best')
plt.show()
