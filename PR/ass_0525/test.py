import numpy as np

a1 = np.array([[1, 2], [3, 4], [5, 6]])
print(a1.shape)
a2 = np.array([[10, 11], [12, 13], [14, 15]])
a3 = np.array([[20, 21], [22, 23], [24, 25]])
a5 = np.array([[a1, a2, a3]])
print(a5)
print(a5.shape,"END\n")

print(np.mean(a5, axis=3))

