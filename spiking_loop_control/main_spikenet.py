#! /usr/bin/env python3

from network.spikenet import SpikeNet

from network.utils import EulerMaruyama, lin_system

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ion()

import pdb

import cProfile, pstats, io
from pstats import SortKey

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sensor_mode", action="store_const", const=True, default=False)

sensor_mode = parser.parse_args().sensor_mode

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
NY = 2 # dimensions of the observation vector y
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

# required for running the dynamical system externally for testing purposes
if sensor_mode:
    u_eff = np.zeros((K))
    integrator = EulerMaruyama(lambda tf, xf: lin_system(xf, u_eff=u_eff, A=A), SIGM_NOISE_D, DT, K)

    u_svd,s_svd,_ = np.linalg.svd(SIGM_NOISE_N)
    P_NOISE_Y = u_svd @ np.diag(np.sqrt(s_svd))

    x_sim = np.array(x0)

net = SpikeNet(N, K, NZ, KZ, sensor_mode=sensor_mode)

'''
A, B, C, D, Dz, l, Time, dt, x0, z0,
                    Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                    SIGM_NOISE_V, SIGM_NOISE_V_Z
'''
net.set_dynamics(A, B, C, D, Dz, l, T_SIM, DT, x0, z_eff[0],
                 Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                 SIGM_NOISE_V, SIGM_NOISE_V_Z)

record_vars = [("lif_pop", "v"),
               ("lif_pop_z", "v"),
               ("lif_pop", "r"),
               ("lif_pop_z", "r")]#,
               #("ds_pop", "x")]

record_spikes = ["lif_pop", "lif_pop_z"]

net.build_network_model(record_vars, record_spikes)

######### manually record x
x_rec = np.ndarray((NT, K))

######################## profiler
pr = cProfile.Profile()
pr.enable()
########################

######################## Run Simulation
for tid in tqdm(range(NT)):
    if sensor_mode:
        u_eff = B @ net.u
        x_sim = integrator.step(tid * DT, x_sim)
        y = C @ x_sim + P_NOISE_Y @ np.random.normal(0., 1., (KZ))

    net.step(tid, z_eff[tid], y)

    if sensor_mode:
        x_rec[tid] = x_sim
    else:
        x_rec[tid] = net.x
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

# v_rec, vz_rec, r_rec, rz_rec, x_rec = net.var_rec_arrays
v_rec, vz_rec, r_rec, rz_rec = net.var_rec_arrays

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
