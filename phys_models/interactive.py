#! /usr/bin/env python3

import numpy as np
import pygame as pg

from .arm2d import Arm2D

from scipy.linalg import solve_continuous_are

from spiking_loop_control.network.spikenet import SpikeNet

import time

L1 = 1.
L2 = 1.
M1 = .25
M2 = .25

I1 = L1**2.*M1/3.
I2 = L2**2.*M2/3.

DAMP1 = .25
DAMP2 = .25

DT = 0.01

##### Network Settings
DT_NETWORK = 0.00005

N_STEPS_SUBSAMPLE = int(DT/DT_NETWORK)
######################## Network Parameters
###### Population
N = 150
K = 4 # size of the dynamical system (?)
NZ = 100 # size of lif z population
KZ = 4 # dimensions of the external target input z
P = 2 # dimensions of control variable u
NY = 4 # dimensions of the observation vector y
######

######################## Additional Parameter Definitions

A = np.zeros((4,4))
A[0,2] = 1.
A[1,3] = 1.
A[2,2] = -(DAMP1+DAMP2)/I1
A[2,3] = DAMP2/I1
A[3,2] = DAMP2/I2
A[3,3] = -DAMP2/I2

B = np.zeros((4,2))
B[2,0] = 1./I1
B[2,1] = -1./I1
B[3,1] = 1./I2

C = np.eye(4) # system readout

D = np.random.randn(K,N) # decoding matrices
D = D / np.sqrt(np.diag(D.T@D)) # normalize vectors
D = D/50. # reduce size

Dz = np.random.randn(KZ,NZ)
Dz = Dz / np.sqrt(np.diag(Dz.T@Dz)) # normalize vectors
Dz = Dz/50. # reduce size

####### noise
SIGM_NOISE_N = 1e-8*np.identity(K)
SIGM_NOISE_D = 1e-8*np.identity(K)
SIGM_NOISE_V = 0e-8 * np.eye(N)
SIGM_NOISE_V_Z = 0e-8 * np.eye(NZ)
#######

####### neuron leakage
l = 10.
#######

####### Kalman filter parameters
Q = np.eye(K)
Q[range(2),range(2)] = 10.
R = 1e-1 * np.eye(P)
#######
########################

arm = Arm2D(L1, L2, M1, M2, np.array([DAMP1, DAMP2]), DT)

rec_arm_state = []
rec_arm_state_net = []

record_spikes = ["lif_pop", "lif_pop_z"]

net = SpikeNet(N, K, NZ, KZ, sensor_mode=True, shared_memory=False)

'''
A, B, C, D, Dz, l, Time, dt, x0, z0,
                    Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                    SIGM_NOISE_V, SIGM_NOISE_V_Z
'''

T_BUFFER_SPIKES = DT_NETWORK * N_STEPS_SUBSAMPLE

net.set_dynamics(A, B, C, D, Dz, l, T_BUFFER_SPIKES, DT_NETWORK, arm.state, arm.targ,
                 Q, R, SIGM_NOISE_N, SIGM_NOISE_D,
                 SIGM_NOISE_V, SIGM_NOISE_V_Z)
net.build_network_model([], record_spikes)

print(arm.state-arm.targ)

# Example file showing a basic pygame "game loop"
import pygame

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

curr_act = np.zeros(2)

mousedown = False

t = 0

t0 = time.time()

r_filt_lif = np.zeros((N))
r_filt_z = np.zeros((NZ))

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

        if event.type == pg.KEYDOWN and event.key == pg.K_UP:
            curr_act[0] += 1.0
        if event.type == pg.KEYUP and event.key == pg.K_UP:
            curr_act[0] -= 1.0
        if event.type == pg.KEYDOWN and event.key == pg.K_DOWN:
            curr_act[0] -= 1.0
        if event.type == pg.KEYUP and event.key == pg.K_DOWN:
            curr_act[0] += 1.0
        if event.type == pg.KEYDOWN and event.key == pg.K_LEFT:
            curr_act[1] += 1.0
        if event.type == pg.KEYUP and event.key == pg.K_LEFT:
            curr_act[1] -= 1.0
        if event.type == pg.KEYDOWN and event.key == pg.K_RIGHT:
            curr_act[1] -= 1.0
        if event.type == pg.KEYUP and event.key == pg.K_RIGHT:
            curr_act[1] += 1.0

        if event.type == pg.MOUSEBUTTONDOWN:
            mousedown = True
        if event.type == pg.MOUSEBUTTONUP:
            mousedown = False

    arm_targ_prev = np.array(arm.targ)

    if mousedown:
        mspos = pg.mouse.get_pos()
        arm.update_ik_targ_angles(np.array([(mspos[0]-CTRX)/ZOOM,-(mspos[1]-CTRY)/ZOOM]))

    arm.step(net.u)
    for _t in range(N_STEPS_SUBSAMPLE):
        #net.step(t*N_STEPS_SUBSAMPLE+_t, arm.targ * l + (arm.targ - arm_targ_prev)/DT, arm.state)
        net.step(t*N_STEPS_SUBSAMPLE+_t, arm.targ * l, arm.state)

    pos_joints = arm.get_joint_pos()

    r_filt_lif -= DT * r_filt_lif * 2.
    r_filt_z -= DT * r_filt_z * 2.
    if T_BUFFER_SPIKES <= DT*t:
        lif_spikes, lif_z_spikes = net.get_spike_recordings()
        r_filt_lif[lif_spikes[1]] += 15.0
        r_filt_z[lif_z_spikes[1]] += 15.0
    r_filt_lif = np.minimum(255., r_filt_lif)
    r_filt_z = np.minimum(255., r_filt_z)

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    # RENDER YOUR GAME HERE

    for k in range(N):
        _x = k%int(np.sqrt(N))
        _y = k//int(np.sqrt(N))
        pg.draw.circle(screen, (0, r_filt_lif[k], r_filt_lif[k]),
                (_x*10+5.,_y*10+5.),5.)
    for k in range(NZ):
        _x = k%int(np.sqrt(NZ)) + np.sqrt(N)
        _y = k//int(np.sqrt(NZ))
        pg.draw.circle(screen, (r_filt_z[k], 0.5 * r_filt_z[k], 0),
                (_x*10+5.+20.,_y*10+5.),5.)

    pg.draw.line(screen, "white",
            (CTRX, CTRY),
            (CTRX + ZOOM*pos_joints[0], CTRY - ZOOM*pos_joints[1]),
            width=2)
    pg.draw.line(screen, "white",
            (CTRX + ZOOM*pos_joints[0], CTRY - ZOOM*pos_joints[1]),
            (CTRX + ZOOM*pos_joints[2], CTRY - ZOOM*pos_joints[3]),
            width=2)
    pg.draw.circle(screen, "white",
            (CTRX + ZOOM*pos_joints[0], CTRY - ZOOM*pos_joints[1]),
            10)
    pg.draw.circle(screen, "white",
            (CTRX + ZOOM*pos_joints[2], CTRY - ZOOM*pos_joints[3]),
            10)


    # flip() the display to put your work on screen
    pygame.display.flip()

    t += 1
    #clock.tick(1./DT)  # limits FPS to 100


    print(f'Real Time / Sim. Time Ratio: {(time.time()-t0)/(t*DT)}', end="\r")

pg.quit()

import matplotlib.pyplot as plt
plt.ion()

rec_arm_state = np.array(rec_arm_state)
rec_arm_state_net = np.array(rec_arm_state_net)

import pdb
pdb.set_trace()
