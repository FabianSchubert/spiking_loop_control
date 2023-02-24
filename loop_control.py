#! /usr/bin/env python3

from pygenn.genn_model import (GeNNModel,
							   create_custom_neuron_class,
							   create_custom_weight_update_class,
							   create_dpf_class,
							   create_custom_init_var_snippet_class,
							   init_var,
							   create_var_ref)

from neurons import Lif, External, Noise
import synapses
from dynsys import damped_spring_mass
from utils import norm_w_no_autapse_model, EulerMaruyama

from scipy.integrate import solve_ivp

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.ion()

import ipdb

######################## Network Parameters
###### Population
N = 200
NDS = 2
######

###### Connections
W_INIT_PARAMS_RECUR = {"mean": 0.0, "sd": 0.5/np.sqrt(N)}
W_INIT_PARAMS_LIF_TO_DS = {"mean": 0.0, "sd": 0.5/np.sqrt(N)}
W_INIT_PARAMS_DS_TO_LIF = {"mean": 0.0, "sd": 0.5/np.sqrt(NDS)}

## Noise
# SIGM_NOISE_LIF is a covariance matrix
# that summarises the effect of both the
# noise term mu_v and the indirect noise term mu_n.
# If the respective covariances are
# C_v, C_n and C_nv (cross-covariance),
# the combined covariance should be
# C_v + F_k C_nv + (F_k C_nv)^T + F C_n F^T
# (If I did the math correctly)

SIGM_NOISE_LIF = np.eye(N)*0.1
# Calculate the projection matrix P_NOISE
# that generates this covariance
# from uncorrelated N-dimensional
# zero-mean noise.
u,s,v = np.linalg.svd(SIGM_NOISE_LIF)
P_NOISE = u @ np.diag(np.sqrt(s))
######

# simulation time step
DT = 0.02
########################

######################## Simulation Parameters
T = 30.0
NT = int(T / DT)
T = NT * DT

t_ax = np.arange(NT) * DT
########################

######################## Dynamical System
x = np.ones((NDS))
u = np.zeros((NDS))

SIGM_NOISE_DS = np.eye((NDS)) * 0.1

f = lambda tf, xf: damped_spring_mass(xf, u=u, k=1.0, g=0.3)

ds_integrator = EulerMaruyama(f, SIGM_NOISE_DS, DT, NDS,
								buffer_rand_samples=NT)
########################

######################## z target
z = np.zeros((NDS))
########################

######################## set up genn model
model = GeNNModel("float", "loop_control")
model.dT = DT
########################

######################## add lif population
lif_pop = model.add_neuron_population(
		"lif_pop",
		N, Lif.model,
		Lif.params,
		Lif.var_init
	)

# enable spike recording for this
# neuron group.
lif_pop.spike_recording_enabled = True
########################

######################## add dynamical system population
ds_pop = model.add_neuron_population(
		"ds_pop",
		NDS, External.model,
		{}, {"x": 0.0, "u": 0.0}
	)
########################

######################## add z population
z_pop = model.add_neuron_population(
		"z_pop",
		NDS, External.model,
		{}, {"x": 0.0, "u": 0.0}
	)
########################

######################## add noise population
noise_pop = model.add_neuron_population(
	"noise_pop",
	N, Noise.model,
	Noise.params, Noise.var_init
	)
########################

######################## add synapse populations
# recurrent fast
W_fast = model.add_synapse_population(
		pop_name="syn_fast",
		source="lif_pop",
		target="lif_pop",
		wu_var_space={"g": init_var(norm_w_no_autapse_model, W_INIT_PARAMS_RECUR)},
		**synapses.fast_syn
	)

# set postsynaptic target for fast synapses
W_fast.ps_target_var = "Isyn_fast"

# recurrent slow
W_slow = model.add_synapse_population(
		pop_name="syn_slow",
		source="lif_pop",
		target="lif_pop",
		wu_var_space={"g": init_var(norm_w_no_autapse_model, W_INIT_PARAMS_RECUR)},
		**synapses.slow_syn
	)

# set postsynaptic target for slow synapses
W_slow.ps_target_var = "Isyn_slow"

# slow synapses for control input u of the dynamical system
D_u = model.add_synapse_population(
		pop_name="syn_lif_to_ds",
		source="lif_pop",
		target="ds_pop",
		wu_var_space={"g": init_var("Normal", W_INIT_PARAMS_LIF_TO_DS)},
		**synapses.slow_syn
	)

# (effective) synapses for input to lif from the dynamical system
FkC = model.add_synapse_population(
	pop_name="syn_ds_to_lif",
	source="ds_pop",
	target="lif_pop",
	wu_var_space={"g": init_var("Uniform", {"min": 0.0, "max": 3.0})},
	**synapses.continuous_external_syn
	)

# set postsynaptic target for ds input synapses
FkC.ps_target_var = "Isyn_ds"

# synapses for input to lif from the (effective) z variable
# (that is, z_eff = dz/dt + lambda * z)
D_z = model.add_synapse_population(
	pop_name="syn_z_to_lif",
	source="z_pop",
	target="lif_pop",
	wu_var_space={"g": init_var("Uniform", {"min": 0.0, "max": 1.0})},
	**synapses.continuous_external_syn
	)

# set postsynaptic target for z input synapses
D_z.ps_target_var = "Isyn_z"

# synapses for noise input to lif - the weights
# determine the noise covariance.
W_noise = model.add_synapse_population(
	pop_name="syn_noise_to_lif",
	source="noise_pop",
	target="lif_pop",
	wu_var_space={"g": P_NOISE.flatten()},
	**synapses.noise_syn
	)

# set postsynaptic target for noise input to lif
W_noise.ps_target_var = "Isyn_noise"
########################

model.build()

# initialise the extra global parameter
# "spikeCount", which is needed for the
# one-spike-at-a-time behaviour.
lif_pop.set_extra_global_param("spikeCount", np.ones(1).astype("int"))

# load the model. num_recording_timesteps
# determines the spike recording buffer size
model.load(num_recording_timesteps=NT)
########################

######################## reference variable views (access to host memory)
v_view = lif_pop.vars["v"].view
x_view = ds_pop.vars["x"].view
u_view = ds_pop.vars["u"].view
z_view = z_pop.vars["x"].view
########################

######################## set up recording arrays (host)
v_rec = np.ndarray((NT, N))
x_rec = np.ndarray((NT, NDS))
u_rec = np.ndarray((NT, NDS))
z_rec = np.ndarray((NT, NDS))
########################

######################## Run Simulation
for tid in tqdm(range(NT)):

	# update model on device
	model.step_time()

	#################### update ds_vars
	ds_pop.pull_var_from_device("u")
	u[:] = u_view

	
	
	x[:] = ds_integrator.step(tid * DT, x)

	x_view[:] = x

	ds_pop.push_var_to_device("x") 
	####################

	#################### update z var
	
	######
	# update your z variable to sth here
	#
	######

	z_view[:] = z
	z_pop.push_var_to_device("x")
	####################

	#################### record stuff if desired
	# In general, pulling data from the
	# device to the host is a bottleneck
	lif_pop.pull_var_from_device("v")
	v_rec[tid] = v_view

	# we already pulled u from the device
	u_rec[tid] = u

	x_rec[tid] = x

	z_rec[tid] = z
	####################

########################

# pull spike recordings
model.pull_recording_buffers_from_device()
lif_spikes = lif_pop.spike_recording_data

fig_sp, ax_sp = plt.subplots(1,1)
ax_sp.plot(lif_spikes[0], lif_spikes[1], '.', c='k', markersize=1)
ax_sp.set_xlabel("$t$")
ax_sp.set_ylabel("Neuron")

fig_ds, ax_ds = plt.subplots(1,1)
ax_ds.plot(t_ax, x_rec)
ax_ds.set_xlabel("$t$")
ax_ds.set_ylabel("$x$")

fig_v, ax_v = plt.subplots(1,1)
ax_v.plot(t_ax, v_rec[:,:5])
ax_v.set_xlabel("$t$")
ax_v.set_ylabel("$v$")

fig_u, ax_u = plt.subplots(1,1)
ax_u.plot(t_ax, u_rec)
ax_u.set_xlabel("$t$")
ax_u.set_ylabel("$u$")

import pdb
pdb.set_trace()