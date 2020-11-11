import matplotlib.pyplot as plt
import numpy as np


# basic config

Wx = 1000
Wy = 1000  # test_area

freq = 1900  # Mhz

# np.random.seed(np.random.uniform(1,))

ddecorr = 200  # shadow
sigmaS = 8  # shadow 标准差 sigmash
deltaS = 0.5

tau4all = 200
tau = 20  # coherence time

BW = 20  # bandwidth

Nf = 9  # Noise figure

kb = 1.381e-23  # Boltzmann constant
T0 = 290  # Noise temp
noisepower = BW * kb * T0 * Nf  # Walt for noise power

rhop = 0.1/noisepower  # channel estimate
rhod = 0.1/noisepower  # dl
rhou = 0.1/noisepower  # ul

# NumFigs = 0
# Nx = 100
# Ny = 100
