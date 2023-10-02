#! /usr/bin/env python3

import numpy as np
import pygame as pg

from arm2d import Arm2D

from scipy.linalg import solve_continuous_are

L1 = 1.
L2 = 1.
M1 = 1.
M2 = 1.

I1 = L1**2.*M1/3.
I2 = L2**2.*M2/3.

DAMP1 = .05
DAMP2 = .05

DT = 0.025

### control stuff

A = np.zeros((4,4))
A[0,2] = 1.
A[1,3] = 1.
A[2,2] = -(DAMP1+DAMP2)/I1
A[2,3] = DAMP2/I1
A[3,2] = DAMP2/I2
A[3,3] = -DAMP2/I2

B = np.zeros((4,2))
B[2,0] = 1.
B[3,1] = 1.

Q = 1.*np.eye(4)
R = 0.1*np.eye(2)

S = solve_continuous_are(A,B,Q,R)
B_i = np.linalg.pinv(B)
R_i = np.linalg.inv(R)
K = R_i @ B.T @ S


arm = Arm2D(L1, L2, M1, M2, np.array([1.,1.]), np.array([DAMP1, DAMP2]), DT)

# Example file showing a basic pygame "game loop"
import pygame

WIDTH = 800
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

arm.state[1] = 1.0

mousedown = False

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

    if mousedown:
        mspos = pg.mouse.get_pos()
        arm.update_ik_targ_angles(np.array([(mspos[0]-CTRX)/ZOOM,-(mspos[1]-CTRY)/ZOOM]))

    #curr_act[0]=arm.atarg[0]-arm.state[0]
    #curr_act[1]=arm.atarg[1]-arm.state[1]
        #arm.state[:2] = arm.atarg
    targ = np.array([arm.atarg[0], arm.atarg[1], 0., 0.])
    curr_act[:] = -K @ (arm.state - targ) - B_i @ targ
    arm.step(curr_act)

    pos_joints = arm.get_joint_pos()

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")

    # RENDER YOUR GAME HERE

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

    clock.tick(60)  # limits FPS to 60

pg.quit()
