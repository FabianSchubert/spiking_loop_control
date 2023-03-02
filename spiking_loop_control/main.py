#! /usr/bin/env python3

from pygenn.genn_model import (GeNNModel,
                               create_custom_neuron_class,
                               create_custom_weight_update_class,
                               create_dpf_class,
                               create_custom_init_var_snippet_class,
                               init_var,
                               create_var_ref)

from network.neurons import Lif, External, Noise
import network.synapses as synapses
from dynsys import lin_system
from utils import EulerMaruyama

from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are as riccati

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ion()

import ipdb

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


########################

np.random.seed(12)

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
z = np.empty([NT+1,K])
z[:int(NT/4)] = np.outer(np.ones(int(NT/4)), np.array([0, 0]))
z[int(NT/4):int(NT/2)] = np.outer(np.ones(int(NT/4)), np.array([1, 0]))
z[int(NT/2):int(3*NT/4)] = np.outer(np.ones(int(NT/4)), np.array([2, 0]))
z[int(3*NT/4):] = np.outer(np.ones(int(NT/4)+1), np.array([1, 0]))

z_dot = np.zeros([K, NT+1])

z_eff = z_dot + l * z
########################

######################## Calculate other Parameters
T = np.diag(D.T@D)/2. # neuron thresholds
Tz = np.diag(Dz.T@Dz)/2.

####### Kalman Filter
X = riccati(A, B, Q, R)
K_r = (B.T@X)/R

Y = riccati(A.T, C, SIGM_NOISE_D, SIGM_NOISE_N)
K_f = Y@C.T@SIGM_NOISE_N
#######

O_A = D.T@(A + l*np.identity(K))@D
O_s = -D.T@D

O_y = -D.T@K_f
O_yr = D.T@K_f@C@D

O_control = -D.T@B@K_r@D
O_z = D.T@B@K_r@Dz

O_r = O_A + O_yr + O_control

SIGM_NOISE_LIF = SIGM_NOISE_V + O_y @ SIGM_NOISE_N @ O_y.T

# Calculate the projection matrix P_NOISE
# that generates this covariance
# from uncorrelated N-dimensional
# zero-mean noise.
u_svd,s_svd,_ = np.linalg.svd(SIGM_NOISE_LIF)
P_NOISE = u_svd @ np.diag(np.sqrt(s_svd))

u_svd,s_svd,_ = np.linalg.svd(SIGM_NOISE_V_Z)
P_NOISE_Z = u_svd @ np.diag(np.sqrt(s_svd))
########################

######################## check if all the matrix dimensions are correct.
# to be done
########################


######################## set up dyn sys variables and integrator
x = np.array(x0)
u_eff = np.zeros((K))


f = lambda tf, xf: lin_system(xf, u_eff=u_eff, A=A)
# note: a reference to the variable u_eff is passed here, not the values.
# This means that changing u_eff later will also affect the result of calling
# f accordingly. Same with A, even though this is not changed anywhere.

ds_integrator = EulerMaruyama(f, SIGM_NOISE_D, DT, K,
                                buffer_rand_samples=NT)
########################

######################## set up genn model
model = GeNNModel("float", "loop_control")
model.dT = DT
########################

######################## "slow synapse rates" initial values
r0 = np.linalg.pinv(D)@x0
rz0 = np.linalg.pinv(Dz)@z[:,0]
########################

######################## add lif population
lif_params = {
    "l": l,
    "vr": 0.0
}

lif_var_init = {
    "v": 0.9*T,
    "vt": T,
    "spike": 0,
    "r": r0
}

lif_pop = model.add_neuron_population(
        "lif_pop",
        N, Lif.model,
        lif_params,
        lif_var_init
    )

# enable spike recording for this
# neuron group.
lif_pop.spike_recording_enabled = True
########################

######################## add lif z population
lif_z_params = {
    "l": l,
    "vr": 0.0,
}

lif_z_var_init = {
    "v": 0.9*Tz,
    "vt": Tz,
    "spike": 0,
    "r": rz0
}

lif_pop_z = model.add_neuron_population(
        "lif_pop_z",
        NZ, Lif.model,
        lif_z_params,
        lif_z_var_init
    )

# enable spike recording for this
# neuron group.
lif_pop_z.spike_recording_enabled = True
########################

######################## add dynamical system population
ds_pop = model.add_neuron_population(
        "ds_pop",
        K, External.model,
        {"l": l},
        {"x": 0.0,
        "u_eff": -B@K_r@D@r0 + B@K_r@Dz@rz0}
    )
########################

######################## add z population
z_eff_pop = model.add_neuron_population(
        "z_eff_pop",
        K, External.model,
        {"l": 0.0}, {"x": 0.0, "u_eff": 0.0}
    )
########################

######################## add noise populations
noise_pop = model.add_neuron_population(
    "noise_pop",
    N, Noise.model,
    Noise.params, Noise.var_init
    )

noise_pop_z = model.add_neuron_population(
    "noise_pop_z",
    NZ, Noise.model,
    Noise.params, Noise.var_init
    )
########################

######################## add synapse populations

# note: when initialising the weights,
# genn wants a flattened version of
# the matrix where w_ij refers to
# i_pre -> j_post, which is the transpose of
# what I consider the usual approach when
# working with weight matrices as operators
# acting on column vectors. Hence, all the weights
# are transposed and flattened before being passed.

########## to lif
# recurrent fast
W_v_v_s = model.add_synapse_population(
        pop_name="syn_lif_to_lif_fast",
        source="lif_pop",
        target="lif_pop",
        wu_var_space={"g": O_s.T.flatten()},
        **synapses.fast_syn
    )

# set postsynaptic target for fast synapses
W_v_v_s.ps_target_var = "Isyn_fast"

# recurrent slow
W_v_v_r = model.add_synapse_population(
        pop_name="syn_lif_to_lif_slow",
        source="lif_pop",
        target="lif_pop",
        wu_var_space={"g": O_r.T.flatten()},
        **synapses.slow_syn,
        ps_param_space={"l": l},
        ps_var_space={"inSynCustom": O_r@r0}
    )

# set postsynaptic target for slow synapses
W_v_v_r.ps_target_var = "Isyn_slow"

# z lif to lif slow
W_v_vz_r = model.add_synapse_population(
    pop_name="syn_lif_z_to_lif_slow",
    source="lif_pop_z",
    target="lif_pop",
    wu_var_space={"g": O_z.T.flatten()},
    **synapses.slow_syn,
    ps_param_space={"l": l},
    ps_var_space={"inSynCustom": O_z@rz0}
    )
W_v_vz_r.ps_target_var = "Isyn_slow"

# (effective) synapses for input to lif from the dynamical system
W_v_x = model.add_synapse_population(
    pop_name="syn_ds_to_lif",
    source="ds_pop",
    target="lif_pop",
    wu_var_space={"g": (O_y@C).T.flatten()},
    **synapses.continuous_external_syn
    )

# set postsynaptic target for ds input synapses
W_v_x.ps_target_var = "Isyn_ds"

# synapses for noise input to lif - the weights
# determine the noise covariance.
W_v_noise = model.add_synapse_population(
    pop_name="syn_noise_to_lif",
    source="noise_pop",
    target="lif_pop",
    wu_var_space={"g": P_NOISE.flatten()},
    **synapses.noise_syn
    )

# set postsynaptic target for noise input to lif
W_v_noise.ps_target_var = "Isyn_noise"
##########

########## to lif z
W_vz_zeff = model.add_synapse_population(
    pop_name="syn_zeff_to_lif_z",
    source="z_eff_pop",
    target="lif_pop_z",
    wu_var_space={"g": (Dz.T).T.flatten()},
    **synapses.continuous_external_syn
    )
W_vz_zeff.ps_target_var = "Isyn_ds"


W_vz_vz_s = model.add_synapse_population(
    pop_name="syn_lif_z_to_lif_z",
    source="lif_pop_z",
    target="lif_pop_z",
    wu_var_space={"g": (-Dz.T@Dz).T.flatten()},
    **synapses.fast_syn
    )
W_vz_vz_s.ps_target_var = "Isyn_fast"

# synapses for noise input to lif z - the weights
# determine the noise covariance.
W_vz_noise = model.add_synapse_population(
    pop_name="syn_noise_to_lif_z",
    source="noise_pop_z",
    target="lif_pop_z",
    wu_var_space={"g": P_NOISE_Z.flatten()},
    **synapses.noise_syn
    )

# set postsynaptic target for noise input to lif
W_vz_noise.ps_target_var = "Isyn_noise"
##########

########## to the dynamical system
# slow synapses from lif for control input u of the dynamical system
W_u_v_r = model.add_synapse_population(
        pop_name="syn_lif_to_ds",
        source="lif_pop",
        target="ds_pop",
        wu_var_space={"g": (-B@K_r@D).T.flatten()},
        **synapses.slow_syn,
        ps_param_space={"l": l},
        ps_var_space={"inSynCustom": -B@K_r@D@r0}
    )
W_u_v_r.ps_target_var = "Isyn"

# slow synapses from lif z for control input u of the dynamical system
W_u_vz_r = model.add_synapse_population(
        pop_name="syn_lif_z_to_ds",
        source="lif_pop_z",
        target="ds_pop",
        wu_var_space={"g": (B@K_r@Dz).T.flatten()},
        **synapses.slow_syn,
        ps_param_space={"l": l},
        ps_var_space={"inSynCustom": B@K_r@Dz@rz0}
    )

W_u_vz_r.ps_target_var = "Isyn"
##########



########################

model.build()

# initialise the extra global parameter
# "spikeCount", which is needed for the
# one-spike-at-a-time behaviour.
lif_pop.set_extra_global_param("spikeCount", np.zeros(1).astype("int"))
lif_pop_z.set_extra_global_param("spikeCount", np.zeros(1).astype("int"))

# load the model. num_recording_timesteps
# determines the spike recording buffer size
model.load(num_recording_timesteps=NT)
########################


######################## reference variable views (access to host memory)
v_view = lif_pop.vars["v"].view
vz_view = lif_pop_z.vars["v"].view
r_view = lif_pop.vars["r"].view
rz_view = lif_pop_z.vars["r"].view
x_view = ds_pop.vars["x"].view
u_eff_view = ds_pop.vars["u_eff"].view
z_eff_view = z_eff_pop.vars["x"].view
########################

######################## set up recording arrays (host)
v_rec = np.ndarray((NT, N))
vz_rec = np.ndarray((NT, NZ))
r_rec = np.ndarray((NT, N))
rz_rec = np.ndarray((NT, NZ))
x_rec = np.ndarray((NT, K))
u_eff_rec = np.ndarray((NT, K))
########################

######################## profiler
pr = cProfile.Profile()
pr.enable()
########################

######################## Run Simulation
for tid in tqdm(range(NT)):

    #################### update ds_vars
    ds_pop.pull_var_from_device("u_eff")
    u_eff[:] = u_eff_view

    x[:] = ds_integrator.step(tid * DT, x)

    x_view[:] = x

    ds_pop.push_var_to_device("x") 
    ####################

    #################### update z var
    z_eff_view[:] = z_eff[:,tid]
    z_eff_pop.push_var_to_device("x")
    ####################

    # update model on device
    model.step_time()

    #'''
    #################### record stuff if desired
    # In general, pulling data from the
    # device to the host is a bottleneck
    lif_pop.pull_var_from_device("v")
    v_rec[tid] = v_view

    lif_pop.pull_var_from_device("r")
    r_rec[tid] = r_view

    lif_pop_z.pull_var_from_device("v")
    vz_rec[tid] = vz_view

    lif_pop_z.pull_var_from_device("r")
    rz_rec[tid] = rz_view

    # we already pulled u_eff from the device
    u_eff_rec[tid] = u_eff

    x_rec[tid] = x
    ####################
    #'''

########################

######################## print profiling stats
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
########################

# pull spike recordings
model.pull_recording_buffers_from_device()
lif_spikes = lif_pop.spike_recording_data
lif_z_spikes = lif_pop_z.spike_recording_data

fig_sp, ax_sp = plt.subplots(1,1)
ax_sp.plot(lif_spikes[0], lif_spikes[1], '.', c='k', markersize=1)
ax_sp.set_xlabel("$t$")
ax_sp.set_ylabel("Neuron")

fig_sp_z, ax_sp_z = plt.subplots(1,1)
ax_sp_z.plot(lif_z_spikes[0], lif_z_spikes[1], '.', c='k', markersize=1)
ax_sp_z.set_xlabel("$t$")
ax_sp_z.set_ylabel("Neuron z")

fig_ds, ax_ds = plt.subplots(1,1)
ax_ds.plot(t_ax, x_rec)
ax_ds.set_xlabel("$t$")
ax_ds.set_ylabel("$x$")

fig_v, ax_v = plt.subplots(1,1)
ax_v.plot(t_ax, v_rec[:,:5])
ax_v.set_xlabel("$t$")
ax_v.set_ylabel("$v$")

fig_vz, ax_vz = plt.subplots(1,1)
ax_vz.plot(t_ax, vz_rec[:,:5])
ax_vz.set_xlabel("$t$")
ax_vz.set_ylabel("vz")

fig_u, ax_u = plt.subplots(1,1)
ax_u.plot(t_ax, u_eff_rec)
ax_u.set_xlabel("$t$")
ax_u.set_ylabel("$u_{eff} = B u$")

import pdb
pdb.set_trace()