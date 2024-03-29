#! /usr/bin/env python3

import numpy as np
import pygame as pg

from phys_models.arm2d import Arm2D

from spiking_loop_control.network.spikenet import SpikeNet

import time

##### arm simulation parameters
L1 = 1.
L2 = 1.
M1 = .25
M2 = .25

I1 = L1**2.*M1/3.
I2 = L2**2.*M2/3.

DAMP1 = .25
DAMP2 = .25

DT = 0.005
#####

##### Network Settings
# supersampling time step for simulating the network.
DT_NETWORK = 0.0001

N_STEPS_SUPERSAMPLE = int(DT/DT_NETWORK)

# only keep spikes buffered for on supersampling
# period (just all spikes appearing in one arm
# simulation time step)
T_BUFFER_SPIKES = DT_NETWORK * N_STEPS_SUPERSAMPLE

######################## Network Parameters
###### Population
N = 150
K = 4 # size of the dynamical system (?)
NZ = 100 # size of lif z population
KZ = 4 # dimensions of the external target input z
P = 2 # dimensions of control variable u
NY = 4 # dimensions of the observation vector y
######
####### neuron leakage
l = 1.
#######
######################## Control Parameter Definitions

A = np.zeros((4,4))
A[0,2] = 1.
A[1,3] = 1.
A[2,2] = -(DAMP1+DAMP2)/I1
A[2,3] = DAMP2/I1
A[3,2] = DAMP2/I2
A[3,3] = -DAMP2/I2

B = np.zeros((K,P))
B[2,0] = 1./I1
B[2,1] = -1./I1
B[3,1] = 1./I2

C = np.eye(4) # system readout
#C = np.zeros((NY,K))
#C[0,0] = 1.
#C[1,1] = 1.

D = np.random.randn(K,N) # decoding matrices
D = D / np.sqrt(np.diag(D.T@D)) # normalize vectors
D = D/50. # reduce size

Dz = np.random.randn(KZ,NZ)
Dz = Dz / np.sqrt(np.diag(Dz.T@Dz)) # normalize vectors
Dz = Dz/50. # reduce size

####### noise
SIGM_NOISE_N = 1e-8*np.identity(NY)
SIGM_NOISE_D = 1e-8*np.identity(K)
SIGM_NOISE_V = 0e-8 * np.eye(N)
SIGM_NOISE_V_Z = 0e-8 * np.eye(NZ)
#######

####### Kalman filter parameters
Q = np.eye(K)
Q[range(2),range(2)] = 10.
R = 2e-2 * np.eye(P)
#######
########################


# create arm instance
arm = Arm2D(L1, L2, M1, M2, np.array([DAMP1, DAMP2]), DT)

# buffer spikes in the network
record_spikes = ["lif_pop", "lif_pop_z"]

# control network instance
net = SpikeNet(N, K, NZ, KZ, sensor_mode=True, shared_memory=False)

net.set_dynamics(A, B, C, D, Dz, l, T_BUFFER_SPIKES, DT_NETWORK, arm.state, arm.targ,
                 Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                 SIGM_NOISE_V, SIGM_NOISE_V_Z)
net.build_network_model([], record_spikes)

# Set up the interactive session.
import pygame

# Window settings
WIDTH = 600
HEIGHT = 600
CTRX = WIDTH//2
CTRY = HEIGHT//2
ZOOM = 100.

# pygame setup
pg.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
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
    r_filt_lif -= DT * r_filt_lif * 2.
    r_filt_z -= DT * r_filt_z * 2.
    if T_BUFFER_SPIKES <= DT*t:
        lif_spikes, lif_z_spikes = net.get_spike_recordings()
        r_filt_lif[lif_spikes[1]] += 15.0
        r_filt_z[lif_z_spikes[1]] += 15.0
    r_filt_lif = np.minimum(255., r_filt_lif)
    r_filt_z = np.minimum(255., r_filt_z)

# draw the arm and the neuronal activity.
def draw_scene():
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    for k in range(N):
        _x = k%int(np.sqrt(N))
        _y = k//int(np.sqrt(N))
        col_interp = 1. - np.exp(-0.01*r_filt_lif[k])
        col = (255*(1.-col_interp), 255*(1.-col_interp) + 100*col_interp, 255)
        pg.draw.circle(screen, col,
                (_x*10+15.,_y*10+15.),5.)
    for k in range(NZ):
        _x = k%int(np.sqrt(NZ)) + np.sqrt(N)
        _y = k//int(np.sqrt(NZ))
        col_interp = 1. - np.exp(-0.01*r_filt_z[k])
        col = (255*(1.-col_interp)+255*col_interp, 255*(1.-col_interp) + 100*col_interp, 255*(1.-col_interp))
        pg.draw.circle(screen, col,
                (_x*10+15.+20.,_y*10+15.),5.)

    pg.draw.line(screen, "black",
            (CTRX, CTRY),
            (CTRX + ZOOM*pos_joints[0], CTRY - ZOOM*pos_joints[1]),
            width=10)
    pg.draw.line(screen, "black",
            (CTRX + ZOOM*pos_joints[0], CTRY - ZOOM*pos_joints[1]),
            (CTRX + ZOOM*pos_joints[2], CTRY - ZOOM*pos_joints[3]),
            width=10)
    pg.draw.circle(screen, "black",
            (CTRX, CTRY), 10)
    pg.draw.circle(screen, "black",
            (CTRX + ZOOM*pos_joints[0], CTRY - ZOOM*pos_joints[1]),
            10)
    pg.draw.circle(screen, "black",
            (CTRX + ZOOM*pos_joints[2], CTRY - ZOOM*pos_joints[3]),
            10)

    mspos = pg.mouse.get_pos()
    pg.draw.circle(screen, "green",
        (mspos[0], mspos[1]), 10)

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
    arm_targ_prev = np.array(arm.targ)

    # update target position / arm state if the user clicks on screen.
    if mousedown:
        mspos = pg.mouse.get_pos()
        arm.update_ik_targ_angles(np.array([(mspos[0]-CTRX)/ZOOM,-(mspos[1]-CTRY)/ZOOM]))
    # update arm state using the current action
    # provided by the network
    arm.step(net.u)

    # run the supersampled network simulation for this time step.
    #net.step(t*N_STEPS_SUPERSAMPLE, arm.targ * l + (arm.targ - arm_targ_prev)/DT, arm.state)
    net.step(t*N_STEPS_SUPERSAMPLE, arm.targ * l, arm.state)
    for _t in range(1, N_STEPS_SUPERSAMPLE-1):
        #net.step(t*N_STEPS_SUPERSAMPLE+_t, arm.targ * l + (arm.targ - arm_targ_prev)/DT, arm.state)
        net.step(t*N_STEPS_SUPERSAMPLE+_t, pull_u=False)
    net.step((t+1)*N_STEPS_SUPERSAMPLE-1, pull_u=True)

    # update caresian coordinates for the joints
    pos_joints = arm.get_joint_pos()

    # update the spike filtering
    update_spike_filters()

    draw_scene()

    # flip backbuffer
    pygame.display.flip()

    t += 1

    # you can limit the simulation speed if it runs faster
    # than realtime.

    pg.image.save(screen, f"./record_frames/frame_{t}.jpg")

    clock.tick(1./DT)  # limits FPS to 1/DT

    print(f'Real Time / Sim. Time Ratio: {(time.time()-t0)/(t*DT)}', end="\r")

pg.quit()

#import matplotlib.pyplot as plt
#plt.ion()

#rec_arm_state = np.array(rec_arm_state)
#rec_arm_state_net = np.array(rec_arm_state_net)

#import pdb
#pdb.set_trace()
