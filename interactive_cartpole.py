#! /usr/bin/env python3

import numpy as np
import pygame as pg

from phys_models.cartpole import Cartpole

from spiking_loop_control.network.spikenet import SpikeNet

import time

##### arm simulation parameters
L1 = 1.0
M1 = 0.5
M2 = 0.25

DAMPX = 0.0  # 25
DAMPTHETA = 0.0  # 25

G = 10.0

DT = 0.005
#####

##### Network Settings
# supersampling time step for simulating the network.
DT_NETWORK = 0.0001

N_STEPS_SUPERSAMPLE = int(DT / DT_NETWORK)

# only keep spikes buffered for on supersampling
# period (just all spikes appearing in one arm
# simulation time step)
T_BUFFER_SPIKES = DT_NETWORK * N_STEPS_SUPERSAMPLE

######################## Network Parameters
###### Population
N = 150
K = 4  # size of the dynamical system (?)
NZ = 100  # size of lif z population
KZ = 4  # dimensions of the external target input z
P = 1  # dimensions of control variable u
NY = 4  # dimensions of the observation vector y
######
####### neuron leakage
l = 1.0
#######
######################## Control Parameter Definitions

# create pole instance
pole = Cartpole(L1, M1, M2, DAMPX, DAMPTHETA, G, DT)

A = pole.A
B = pole.B

C = np.eye(4)  # system readout
# C = np.zeros((NY,K))
# C[0,0] = 1.
# C[1,1] = 1.

D = np.random.randn(K, N)  # decoding matrices
D = D / np.sqrt(np.diag(D.T @ D))  # normalize vectors
D = D / 50.0  # reduce size

Dz = np.random.randn(KZ, NZ)
Dz = Dz / np.sqrt(np.diag(Dz.T @ Dz))  # normalize vectors
Dz = Dz / 50.0  # reduce size

####### noise
SIGM_NOISE_N = 1e-8 * np.identity(NY)
SIGM_NOISE_D = 1e-8 * np.identity(K)
SIGM_NOISE_V = 0e-8 * np.eye(N)
SIGM_NOISE_V_Z = 0e-8 * np.eye(NZ)
#######

####### Kalman filter parameters
Q = np.eye(K)
Q[range(2), range(2)] = 10.0
R = 2e-2 * np.eye(P)
#######
########################

# buffer spikes in the network
record_spikes = ["lif_pop", "lif_pop_z"]

# control network instance
net = SpikeNet(N, K, NZ, KZ, sensor_mode=True, shared_memory=False)

net.set_dynamics(
    A,
    B,
    C,
    D,
    Dz,
    l,
    T_BUFFER_SPIKES,
    DT_NETWORK,
    pole.state,
    pole.targ,
    Q,
    R,
    SIGM_NOISE_N,
    SIGM_NOISE_D,
    SIGM_NOISE_V,
    SIGM_NOISE_V_Z,
)
net.build_network_model([], record_spikes)

# Set up the interactive session.

# Window settings
WIDTH = 600
HEIGHT = 600
CTRX = WIDTH // 2
CTRY = HEIGHT // 2
ZOOM = 100.0

# pygame setup
pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))
clock = pg.time.Clock()
running = True
mousedown = False

# keep track of time
t = 0

# starting time
t0 = time.time()

# variables for temporal filters
# of the spiking activity.
r_filt_lif = np.zeros((N))
r_filt_z = np.zeros((NZ))


# update the linear filter of
# spike activity
def update_spike_filters():
    global r_filt_lif, r_filt_z
    r_filt_lif -= DT * r_filt_lif * 2.0
    r_filt_z -= DT * r_filt_z * 2.0
    if T_BUFFER_SPIKES <= DT * t:
        lif_spikes, lif_z_spikes = net.get_spike_recordings()
        r_filt_lif[lif_spikes[1]] += 15.0
        r_filt_z[lif_z_spikes[1]] += 15.0
    r_filt_lif = np.minimum(255.0, r_filt_lif)
    r_filt_z = np.minimum(255.0, r_filt_z)


# draw the arm and the neuronal activity.
def draw_scene():
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    for k in range(N):
        _x = k % int(np.sqrt(N))
        _y = k // int(np.sqrt(N))
        col_interp = 1.0 - np.exp(-0.01 * r_filt_lif[k])
        col = (
            255 * (1.0 - col_interp),
            255 * (1.0 - col_interp) + 100 * col_interp,
            255,
        )
        pg.draw.circle(screen, col, (_x * 10 + 15.0, _y * 10 + 15.0), 5.0)
    for k in range(NZ):
        _x = k % int(np.sqrt(NZ)) + np.sqrt(N)
        _y = k // int(np.sqrt(NZ))
        col_interp = 1.0 - np.exp(-0.01 * r_filt_z[k])
        col = (
            255 * (1.0 - col_interp) + 255 * col_interp,
            255 * (1.0 - col_interp) + 100 * col_interp,
            255 * (1.0 - col_interp),
        )
        pg.draw.circle(screen, col, (_x * 10 + 15.0 + 20.0, _y * 10 + 15.0), 5.0)

    pos_base = pole.base_pos
    pos_tip = pole.tip_pos

    pg.draw.line(
        screen,
        "black",
        (CTRX + ZOOM * pos_base[0], CTRY - ZOOM * pos_base[1]),
        (CTRX + ZOOM * pos_tip[0], CTRY - ZOOM * pos_tip[1]),
        width=10,
    )

    pg.draw.circle(
        screen, "black", (CTRX + ZOOM * pos_base[0], CTRY - ZOOM * pos_base[1]), 10
    )

    pg.draw.circle(
        screen, "black", (CTRX + ZOOM * pos_tip[0], CTRY - ZOOM * pos_tip[1]), 10
    )

    mspos = pg.mouse.get_pos()
    pg.draw.circle(screen, "green", (mspos[0], mspos[1]), 10)


while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEBUTTONDOWN:
            mousedown = True
        if event.type == pg.MOUSEBUTTONUP:
            mousedown = False

    # for calculating the temporal derivative of the target.
    pole_targ_prev = np.array(pole.targ)

    # update target position / pole state if the user clicks on screen.
    if mousedown:
        mspos = pg.mouse.get_pos()
        pole._targ[0] = (mspos[0] - CTRX) / ZOOM

    # update arm state using the current action
    # provided by the network
    pole.step(net.u)

    # run the supersampled network simulation for this time step.
    # net.step(t*N_STEPS_SUPERSAMPLE, arm.targ * l + (arm.targ - arm_targ_prev)/DT, arm.state)
    net.step(t * N_STEPS_SUPERSAMPLE, pole.targ * l, pole.state)
    for _t in range(1, N_STEPS_SUPERSAMPLE - 1):
        # net.step(t*N_STEPS_SUPERSAMPLE+_t, arm.targ * l + (arm.targ - arm_targ_prev)/DT, arm.state)
        net.step(t * N_STEPS_SUPERSAMPLE + _t, pull_u=False)
    net.step((t + 1) * N_STEPS_SUPERSAMPLE - 1, pull_u=True)

    # update the spike filtering
    update_spike_filters()

    draw_scene()

    # flip backbuffer
    pg.display.flip()

    t += 1

    # you can limit the simulation speed if it runs faster
    # than realtime.

    # pg.image.save(screen, f"./record_frames/frame_{t}.jpg")

    clock.tick(1.0 / DT)  # limits FPS to 1/DT

    print(f"Real Time / Sim. Time Ratio: {(time.time()-t0)/(t*DT)}", end="\r")

pg.quit()

# import matplotlib.pyplot as plt
# plt.ion()

# rec_arm_state = np.array(rec_arm_state)
# rec_arm_state_net = np.array(rec_arm_state_net)

# import pdb
# pdb.set_trace()
