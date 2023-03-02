#! /usr/bin/env python3

from network.spikenet import SpikeNet

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ion()

import pdb

import cProfile, pstats, io
from pstats import SortKey

######################## Simulation Parameters
T_SIM = 20.0
# simulation time step
DT = 0.0001
NT = int(T_SIM / DT)
T_SIM = NT * DT

t_ax = np.arange(NT) * DT
########################

######################## Network Parameters
###### Population
N = 100
K = 2 # size of the dynamical system (?)
NZ = 50 # size of lif z population
KZ = 2 # dimensions of the external control input z
P = 1 # dimensions of control variable u
NY = 2 # dimensions of the (implicit) observation vector y
######

######################## Additional Parameter Definitions
######## lin system + control
m = 20.
k = 6.
c = 2.

w = np.sqrt(k/m)
damp = c/(2 * m * w)

A = np.array([[0.0, 1.0], # linear system coupling
              [-w**2, -2*damp*w]])

B = np.array([[0.0, 1.0]]).T # control input

C = np.array([[1.0, 0.0],
              [0.0, 0.0]]) # system readout

D = np.random.randn(K,N) # decoding matrices
D = D / np.sqrt(np.diag(D.T@D)) # normalize vectors
D = D/50. # reduce size

Dz = np.random.randn(KZ,NZ)
Dz = Dz / np.sqrt(np.diag(Dz.T@Dz)) # normalize vectors
Dz = Dz/50. # reduce size

####### noise
SIGM_NOISE_N = 1e-4*np.identity(K)
SIGM_NOISE_D = 1e-4*np.identity(K)
SIGM_NOISE_V = np.eye(N)*0.0
SIGM_NOISE_V_Z = np.eye(NZ)*0.0
#######

####### neuron leakage
l = 1.0
#######

####### Kalman filter parameters
Q = np.identity(K)
Q[0, 0] = 10.0
R = 1e-2
#######
########################

# init x
x0 = np.array([3.0, 0.0])

######################## Control Signal
z = np.empty([NT,K])
z[:int(NT/4)] = np.outer(np.ones(int(NT/4)), np.array([0, 0]))
z[int(NT/4):int(NT/2)] = np.outer(np.ones(int(NT/4)), np.array([1, 0]))
z[int(NT/2):int(3*NT/4)] = np.outer(np.ones(int(NT/4)), np.array([2, 0]))
z[int(3*NT/4):] = np.outer(np.ones(int(NT/4)), np.array([1, 0]))

z_dot = np.zeros([NT,K])

z_eff = z_dot + l * z
########################

net = SpikeNet(N, K, NZ, KZ)

'''
A, B, C, D, Dz, l, Time, dt, x0, z0,
                    Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                    SIGM_NOISE_V, SIGM_NOISE_V_Z
'''
net.set_dynamics(A, B, C, D, Dz, l, T_SIM, DT, x0, z[0],
                 Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                 SIGM_NOISE_V, SIGM_NOISE_V_Z)

record_vars = [("lif_pop", "v"),
               ("lif_pop_z", "v"),
               ("lif_pop", "r"),
               ("lif_pop_z", "r"),
               ("ds_pop", "x")]

net.build_network_model(record_vars, ["lif_pop", "lif_pop_z"])

######################## profiler
pr = cProfile.Profile()
pr.enable()
########################

######################## Run Simulation
for tid in tqdm(range(NT)):
    net.step(tid, z[tid])
########################

######################## print profiling stats
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
########################

lif_spikes, lif_z_spikes = net.get_spike_recordings()

v_rec, vz_rec, r_rec, rz_rec, x_rec = net.var_rec_arrays

fig_sp, ax_sp = plt.subplots(1,1)
ax_sp.plot(lif_spikes[0], lif_spikes[1], '.', c='k', markersize=1)
ax_sp.set_xlabel("$t$")
ax_sp.set_ylabel("Neuron")
fig_sp.tight_layout()

fig_sp_z, ax_sp_z = plt.subplots(1,1)
ax_sp_z.plot(lif_z_spikes[0], lif_z_spikes[1], '.', c='k', markersize=1)
ax_sp_z.set_xlabel("$t$")
ax_sp_z.set_ylabel("Neuron z")
fig_sp_z.tight_layout()

fig_ds, ax_ds = plt.subplots(1,1)
ax_ds.plot(t_ax, x_rec)
ax_ds.set_xlabel("$t$")
ax_ds.set_ylabel("$x$")
fig_ds.tight_layout()

fig_v, ax_v = plt.subplots(1,1)
ax_v.plot(t_ax, v_rec[:,0])
ax_v.set_xlabel("$t$")
ax_v.set_ylabel("$v$")
fig_v.tight_layout()

fig_vz, ax_vz = plt.subplots(1,1)
ax_vz.plot(t_ax, vz_rec[:,0])
ax_vz.set_xlabel("$t$")
ax_vz.set_ylabel("vz")
fig_vz.tight_layout()

plt.show()

plt.ion()
pdb.set_trace()