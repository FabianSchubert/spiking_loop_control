#! /usr/bin/env python3

from pygenn.genn_model import (GeNNModel,
                               create_custom_neuron_class,
                               create_custom_weight_update_class,
                               create_dpf_class,
                               create_custom_init_var_snippet_class,
                               init_var,
                               create_var_ref,
                               create_custom_custom_update_class,
                               create_egp_ref)

from pygenn.genn_wrapper import VarLocation_HOST_DEVICE_ZERO_COPY

from .neurons import Lif, External, Noise
from . import synapses
from .utils import EulerMaruyama, lin_system

from scipy.linalg import solve_continuous_are as riccati

import numpy as np

spikeCount_custom_update_model = create_custom_custom_update_class(
    "spikeCount_update_model",
    var_refs=[("x", "scalar")], param_names=[],
    extra_global_param_refs=[("spikeCount", "int*")],
    update_code="""
        *($(spikeCount)) = 0;
    """
)

class SpikeNet:

    def __init__(self, N, K, NZ, KZ, sensor_mode=False, shared_memory=False):
        '''
            N: Dimension of the spiking sontrol network
            K: Dimension of the dynamical system to be controlled.
            NZ: Dimension of the spiking network encoding the target variable (z)
            KZ: Dimension of the target / obversable.
        '''
        self.N = N
        self.K = K
        self.NZ = NZ
        self.KZ = KZ
        self.sensor_mode = sensor_mode
        self.shared_memory = shared_memory

    def set_dynamics(self, A, B, C, D, Dz, l, Time, dt, x0, z0,
                    Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                    SIGM_NOISE_V, SIGM_NOISE_V_Z):
        '''
            A, B: Matrices defining the dynamical system dx(t)/dt = Ax(t) + Bu(t) + psi_d(t)
            C: Readout matrix defining the observable y(t) = Cx(t) + psi_n(t)
            D: Decoding matrix
            D: Decoding matrix z population
        '''
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Dz = Dz

        self.NU = self.B.shape[1] # dimension of the control variable u.

        self.T = np.diag(self.D.T@self.D)/2. # neuron thresholds
        self.Tz = np.diag(self.Dz.T@self.Dz)/2.

        self.DT = dt
        self.T_SIM = Time
        self.NT = int(Time/dt)
        self.T_SIM = self.NT * self.DT
        self.l = l
        self.x0 = x0
        self.z0 = z0

        self.Q = Q
        self.R = R

        self.SIGM_NOISE_N = SIGM_NOISE_N
        self.SIGM_NOISE_D = SIGM_NOISE_D
        self.SIGM_NOISE_V = SIGM_NOISE_V
        self.SIGM_NOISE_V_Z = SIGM_NOISE_V_Z

        ####### Kalman Filter
        self.X = riccati(self.A, self.B, self.Q, self.R)
        self.K_r = (self.B.T@self.X)/self.R

        self.Y = riccati(self.A.T, self.C, self.SIGM_NOISE_N, self.SIGM_NOISE_D)
        self.K_f = self.Y@self.C.T@self.SIGM_NOISE_D
        #######

        self.O_A = self.D.T@(self.A + self.l*np.identity(self.K))@self.D
        self.O_s = -self.D.T@self.D

        self.O_y = -self.D.T@self.K_f
        self.O_yr = self.D.T@self.K_f@self.C@self.D

        self.O_control = -self.D.T@self.B@self.K_r@self.D
        self.O_z = self.D.T@self.B@self.K_r@self.Dz

        self.O_r = self.O_A + self.O_yr + self.O_control

        self.SIGM_NOISE_LIF = self.SIGM_NOISE_V
        # self.SIGM_NOISE_LIF = self.SIGM_NOISE_V + self.O_y @ self.SIGM_NOISE_N @ self.O_y.T

        # Calculate the projection matrix P_NOISE
        # that generates this covariance
        # from uncorrelated N-dimensional
        # zero-mean noise.
        u_svd,s_svd,_ = np.linalg.svd(self.SIGM_NOISE_LIF)
        self.P_NOISE = u_svd @ np.diag(np.sqrt(s_svd))

        u_svd,s_svd,_ = np.linalg.svd(self.SIGM_NOISE_V_Z)
        self.P_NOISE_Z = u_svd @ np.diag(np.sqrt(s_svd))

        if not self.sensor_mode:
            u_svd,s_svd,_ = np.linalg.svd(self.SIGM_NOISE_N)
            self.P_NOISE_Y = u_svd @ np.diag(np.sqrt(s_svd))

        self.x = np.array(self.x0)
        self.y = np.zeros(self.KZ)
        self.u = np.zeros(self.NU)
        self.u_eff = np.zeros((self.K))

        if not self.sensor_mode:
            self.f = lambda tf, xf: lin_system(xf, u_eff=self.u_eff, A=self.A)
           # note: a reference to the variable u_eff is passed here, not the values.
            # This means that changing u_eff later will also affect the result of calling
            # f accordingly. Same with A, even though this is not changed anywhere.

            self.ds_integrator = EulerMaruyama(self.f, self.SIGM_NOISE_D, self.DT, self.K,
                                               buffer_rand_samples=self.NT)

        ######################## "slow synapse rates" initial values
        self.r0 = np.linalg.pinv(self.D)@self.x0
        self.rz0 = np.linalg.pinv(self.Dz)@self.z0
        ########################

    def build_network_model(self, record_variables, record_spikes):

        self.record_variables = record_variables
        self.record_spikes = record_spikes

        ######################## set up genn model
        self.model = GeNNModel("float", "loop_control")
        self.model.dT = self.DT
        ########################

        self.add_neuron_populations()
        self.add_synapse_populations()

        self.spikeCount_custom_update_lif = self.model.add_custom_update(
            "spikeCount_custom_update_lif", "spike_count_update",
            spikeCount_custom_update_model, {}, {}, {"x": create_var_ref(self.lif_pop, "v")},
            {"spikeCount": create_egp_ref(self.lif_pop, "spikeCount")}
        )

        self.spikeCount_custom_update_lif_z = self.model.add_custom_update(
            "spikeCount_custom_update_lif_z", "spike_count_update",
            spikeCount_custom_update_model, {}, {}, {"x": create_var_ref(self.lif_pop_z, "v")},
            {"spikeCount": create_egp_ref(self.lif_pop_z, "spikeCount")}
        )

        if self.shared_memory:
            self.u_pop.pop.set_var_location("u_eff", VarLocation_HOST_DEVICE_ZERO_COPY)
            self.y_pop.pop.set_var_location("x", VarLocation_HOST_DEVICE_ZERO_COPY)
            self.z_eff_pop.pop.set_var_location("x", VarLocation_HOST_DEVICE_ZERO_COPY)

        for pop in self.record_spikes:
            self.model.neuron_populations[pop].spike_recording_enabled = True

        self.model.build()
        
        # initialise the extra global parameter
        # "spikeCount", which is needed for the
        # one-spike-at-a-time behaviour.
        self.lif_pop.set_extra_global_param("spikeCount", np.zeros(1).astype("int"))
        self.lif_pop_z.set_extra_global_param("spikeCount", np.zeros(1).astype("int"))
        
        # load the model. num_recording_timesteps
        # determines the spike recording buffer size
        self.model.load(num_recording_timesteps=self.NT)

        self.var_views = []
        self.var_rec_arrays = []

        for pop, var in self.record_variables:
            self.var_views.append(self.model.neuron_populations[pop].vars[var].view)
            self.var_rec_arrays.append(np.ndarray((self.NT, self.model.neuron_populations[pop].size)))

        # self.u_eff_view = self.ds_pop.vars["u_eff"].view
        self.u_view = self.u_pop.vars["u_eff"].view
        # self.x_view = self.ds_pop.vars["x"].view
        self.y_view = self.y_pop.vars["x"].view
        self.z_eff_view = self.z_eff_pop.vars["x"].view

    def add_neuron_populations(self):
        ######################## add lif population
        lif_params = {
            "l": self.l,
            "vr": 0.0
        }

        lif_var_init = {
            "v": 0.9*self.T,
            "vt": self.T,
            "spike": 0,
            "r": self.r0
        }

        self.lif_pop = self.model.add_neuron_population(
                "lif_pop",
                self.N, Lif.model,
                lif_params,
                lif_var_init
            )
        ########################

        ######################## add lif z population
        lif_z_params = {
            "l": self.l,
            "vr": 0.0,
        }

        lif_z_var_init = {
            "v": 0.9*self.Tz,
            "vt": self.Tz,
            "spike": 0,
            "r": self.rz0
        }

        self.lif_pop_z = self.model.add_neuron_population(
                "lif_pop_z",
                self.NZ, Lif.model,
                lif_z_params,
                lif_z_var_init
            )
        ########################

        ######################## add y state population
        self.y_pop = self.model.add_neuron_population(
            "y_pop",
            self.KZ, External.model,
            {"l": 0.0}, {"x": 0.0, "u_eff": 0.0}
        )
        ########################

        ######################## add u state population
        self.u_pop = self.model.add_neuron_population(
            "u_pop",
            self.NU, External.model,
            {"l": 0.0}, {"x": 0.0, "u_eff": 0.0}
        )
        ########################

        '''
        ######################## add dynamical system population
        self.ds_pop = self.model.add_neuron_population(
                "ds_pop",
                self.K, External.model,
                {"l": self.l},
                {"x": 0.0,
                "u_eff": self.B@self.K_r@(self.Dz@self.rz0-self.D@self.r0)}
            )
        ########################
        '''

        ######################## add z population
        self.z_eff_pop = self.model.add_neuron_population(
                "z_eff_pop",
                self.K, External.model,
                {"l": 0.0}, {"x": 0.0, "u_eff": 0.0}
            )
        ########################

        ######################## add noise populations
        self.noise_pop = self.model.add_neuron_population(
            "noise_pop",
            self.N, Noise.model,
            Noise.params, Noise.var_init
            )

        self.noise_pop_z = self.model.add_neuron_population(
            "noise_pop_z",
            self.NZ, Noise.model,
            Noise.params, Noise.var_init
            )
        ########################


    def add_synapse_populations(self):
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
        self.W_v_v_s = self.model.add_synapse_population(
                pop_name="syn_lif_to_lif_fast",
                source="lif_pop",
                target="lif_pop",
                wu_var_space={"g": self.O_s.T.flatten()},
                **synapses.fast_syn
            )

        # set postsynaptic target for fast synapses
        self.W_v_v_s.ps_target_var = "Isyn_fast"

        # recurrent slow
        self.W_v_v_r = self.model.add_synapse_population(
                pop_name="syn_lif_to_lif_slow",
                source="lif_pop",
                target="lif_pop",
                wu_var_space={"g": self.O_r.T.flatten()},
                **synapses.slow_syn,
                ps_param_space={"l": self.l},
                ps_var_space={"inSynCustom": self.O_r@self.r0}
            )

        # set postsynaptic target for slow synapses
        self.W_v_v_r.ps_target_var = "Isyn_slow"

        # z lif to lif slow
        self.W_v_vz_r = self.model.add_synapse_population(
            pop_name="syn_lif_z_to_lif_slow",
            source="lif_pop_z",
            target="lif_pop",
            wu_var_space={"g": self.O_z.T.flatten()},
            **synapses.slow_syn,
            ps_param_space={"l": self.l},
            ps_var_space={"inSynCustom": self.O_z@self.rz0}
            )
        self.W_v_vz_r.ps_target_var = "Isyn_slow"

        '''
        # (effective) synapses for input to lif from the dynamical system
        self.W_v_x = self.model.add_synapse_population(
            pop_name="syn_ds_to_lif",
            source="ds_pop",
            target="lif_pop",
            wu_var_space={"g": (self.O_y@self.C).T.flatten()},
            **synapses.continuous_external_syn
            )

        # set postsynaptic target for ds input synapses
        self.W_v_x.ps_target_var = "Isyn_ds"
        '''

        # synapses for input to lif from the observations y
        self.W_v_y = self.model.add_synapse_population(
            pop_name="syn_y_to_lif",
            source="y_pop",
            target="lif_pop",
            wu_var_space={"g": self.O_y.T.flatten()},
            **synapses.continuous_external_syn
        )

        # set postsynaptic target for y input synapses
        self.W_v_y.ps_target_var = "Isyn_ds"

        # synapses for noise input to lif - the weights
        # determine the noise covariance.
        self.W_v_noise = self.model.add_synapse_population(
            pop_name="syn_noise_to_lif",
            source="noise_pop",
            target="lif_pop",
            wu_var_space={"g": self.P_NOISE.flatten()},
            **synapses.noise_syn
            )

        # set postsynaptic target for noise input to lif
        self.W_v_noise.ps_target_var = "Isyn_noise"
        ##########

        ########## to lif z
        self.W_vz_zeff = self.model.add_synapse_population(
            pop_name="syn_zeff_to_lif_z",
            source="z_eff_pop",
            target="lif_pop_z",
            wu_var_space={"g": (self.Dz.T).T.flatten()},
            **synapses.continuous_external_syn
            )
        self.W_vz_zeff.ps_target_var = "Isyn_ds"


        self.W_vz_vz_s = self.model.add_synapse_population(
            pop_name="syn_lif_z_to_lif_z",
            source="lif_pop_z",
            target="lif_pop_z",
            wu_var_space={"g": (-self.Dz.T@self.Dz).T.flatten()},
            **synapses.fast_syn
            )

        self.W_vz_vz_s.ps_target_var = "Isyn_fast"

        # synapses for noise input to lif z - the weights
        # determine the noise covariance.
        self.W_vz_noise = self.model.add_synapse_population(
            pop_name="syn_noise_to_lif_z",
            source="noise_pop_z",
            target="lif_pop_z",
            wu_var_space={"g": self.P_NOISE_Z.flatten()},
            **synapses.noise_syn
            )

        # set postsynaptic target for noise input to lif
        self.W_vz_noise.ps_target_var = "Isyn_noise"
        ##########

        ########## to the control variable u
        # slow synapses from lif for control input u of the dynamical system
        self.W_u_v_r = self.model.add_synapse_population(
                pop_name="syn_lif_to_u",
                source="lif_pop",
                target="u_pop",
                wu_var_space={"g": (-self.K_r@self.D).T.flatten()},
                **synapses.slow_syn,
                ps_param_space={"l": self.l},
                ps_var_space={"inSynCustom": -self.K_r@self.D@self.r0}
            )
        self.W_u_v_r.ps_target_var = "Isyn"

        # slow synapses from lif z for control input u of the dynamical system
        self.W_u_vz_r = self.model.add_synapse_population(
                pop_name="syn_lif_z_to_u",
                source="lif_pop_z",
                target="u_pop",
                wu_var_space={"g": (self.K_r@self.Dz).T.flatten()},
                **synapses.slow_syn,
                ps_param_space={"l": self.l},
                ps_var_space={"inSynCustom": self.K_r@self.Dz@self.rz0}
            )

        self.W_u_vz_r.ps_target_var = "Isyn"
        ##########

    def step(self, tid, z, y=None):

        #################### update ds_vars
        self.u_pop.pull_var_from_device("u_eff")
        self.u[:] = self.u_view

        if not self.sensor_mode:
            self.u_eff = self.B @ self.u
            self.x[:] = self.ds_integrator.step(tid * self.DT, self.x)
            self.y_view[:] = self.C @ self.x + self.P_NOISE_Y @ np.random.normal(0., 1., (self.KZ))
        else:
            self.y_view[:] = y

        self.y_pop.push_var_to_device("x")
        ####################

        #################### update z var
        self.z_eff_view[:] = z
        self.z_eff_pop.push_var_to_device("x")
        ####################

        # update model on device
        self.model.step_time()

        # update spikeCounter extra global parameters
        self.model.custom_update("spike_count_update")
        # self.lif_pop.extra_global_params["spikeCount"].view[:] = 0
        # self.lif_pop.push_extra_global_param_to_device("spikeCount", 1)

        # self.lif_pop_z.extra_global_params["spikeCount"].view[:] = 0
        # self.lif_pop_z.push_extra_global_param_to_device("spikeCount", 1)

        for k, (pop, var) in enumerate(self.record_variables):
            self.model.neuron_populations[pop].pull_var_from_device(var)
            self.var_rec_arrays[k][tid] = self.var_views[k]

    def get_spike_recordings(self):

        self.model.pull_recording_buffers_from_device()

        return [self.model.neuron_populations[pop].spike_recording_data for pop in self.record_spikes]
