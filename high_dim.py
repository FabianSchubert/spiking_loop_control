#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from spiking_loop_control.network.spikenet import SpikeNet


# Make Dynamics Matrix
def make_A(N, k, c):
    A = np.zeros([2 * (N + 1), 2 * (N + 1)])
    for i in range(N):
        A[2 * i, 2 * i + 1] = 1
        A[2 * i + 1, 2 * i + 1] = -c

        A[2 * i + 1, 2 * i] = -k[i] - k[i + 1]
        A[2 * i + 1, 2 * (i - 1)] = k[i]
        A[2 * i + 1, 2 * (i + 1)] = k[i + 1]
    A[1, -2] = 0
    return A


# Make Starting Vector
def make_start(N, L):
    start_pos = np.linspace(0, L, N + 1)
    start_vec = np.empty(2 * (N + 1))
    for i in range(N):
        start_vec[2 * i] = start_pos[i]
        start_vec[2 * i + 1] = 0
    start_vec[-2] = L
    start_vec[-1] = 0
    return start_vec


# Make plots from the simulation
def plots(state_vec, E):
    for i in range(N + 1):
        plt.plot(time_vec, state_vec[2 * i, :])
    plt.show()

    plt.plot(time_vec, E)
    plt.ylim(0, 1.5 * E[0])
    plt.show()


# set up sim parameters
T = 10.0
DT = 1e-4
NT = int(T / DT)
time_vec = np.arange(NT) * DT

# set up physical parameters
N = 100
L = 100
C_SYSTEM = 0.01
rng = np.random.default_rng()
K_SYSTEM = 1.0 + rng.exponential(scale=2, size=N + 1)
A = make_A(N, K_SYSTEM, C_SYSTEM)
start_vec = make_start(N, L)

# create spiking network
K = 2 * (N + 1)
N = 500

####### noise
LAMBD = 1.0

P = K
NZ = 500
KZ = K
NY = K

SIGM_NOISE_N = 1e-8 * np.identity(NY)
SIGM_NOISE_D = 1e-8 * np.identity(K)
SIGM_NOISE_V = 0e-8 * np.eye(N)
SIGM_NOISE_V_Z = 0e-8 * np.eye(NZ)

D = np.random.randn(K, N)
D = D / np.sqrt(np.diag(D.T @ D))

Dz = np.random.randn(KZ, NZ)
Dz = Dz / np.sqrt(np.diag(Dz.T @ Dz))  # normalize vectors
Dz = Dz / 50.0  # reduce size

Q = np.eye(K)
# Q[range(2),range(2)] = 10.
R = 2e-2 * np.eye(P)

B = np.eye(K)
C = np.eye(K)

record_spikes = ["lif_pop", "lif_pop_z"]

net = SpikeNet(N, K, NZ, KZ)
net.set_dynamics(
    A,
    B,
    C,
    D,
    Dz,
    LAMBD,
    T,
    DT,
    start_vec,
    np.zeros((KZ)),
    Q,
    R,
    SIGM_NOISE_N,
    SIGM_NOISE_D,
    SIGM_NOISE_V,
    SIGM_NOISE_V_Z,
)
net.build_network_model([], record_spikes)

for t in tqdm(range(NT)):
    net.step(t)
