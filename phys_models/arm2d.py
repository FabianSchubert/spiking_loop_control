import numpy as np


class Arm2D:

    def __init__(self, l1, l2, m1, m2, torque_mag, damp, dt):

        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.I1 = self.m1 * self.l1**2. / 3.
        self.I2 = self.m2 * self.l2**2. / 3.
        self.torque_mag = torque_mag # 2d array with torques applied when active
        self.damp = damp # velocity damping on both rods
        self.dt = dt # time step

        self.state = np.zeros((4))

        self.atarg = np.zeros((2))

    def step(self, action):

        angles = self.state[:2]
        vels = self.state[2:]

        total_torques = action * self.torque_mag

        dadt = vels

        #diffa = angles[1] - angles[0]
        #dv1 = -(self.l2 * self.m2 * np.sin(diffa) * vels[1]**2. + self.l1 * self.m2 * np.cos(diffa) * np.sin(diffa) * vels[0]**2.)
        #dv1 /= self.l1 * self.m2 * np.cos(diffa)**2. - self.l1 * (self.m1 + self.m2)
        dv1 = 0.0

        #dv2 = self.l2 * self.m2 * np.cos(diffa) * np.sin(diffa) * vels[1]**2.  + np.sin(diffa) * self.l1 * vels[0]**2. * (self.m1 + self.m2)
        #dv2 /= self.l2 * self.m2 * np.cos(diffa)**2. - self.l2 * (self.m1 + self.m2)
        dv2 = 0.0

        dv1 += (total_torques[0] - total_torques[1]) / self.I1
        dv2 += total_torques[1] / self.I2

        dv1 -= (vels[0] * self.damp[0] + (vels[0] - vels[1]) * self.damp[1]) / self.I1
        dv2 -= (vels[1] - vels[0]) * self.damp[1] / self.I2

        self.state[:2] += self.dt * dadt
        self.state[2:] += self.dt * np.array([dv1, dv2])

    def get_joint_pos(self):
        return np.array([self.l1 * np.cos(self.state[0]), self.l1 * np.sin(self.state[0]),
                         self.l1 * np.cos(self.state[0]) + self.l2 * np.cos(self.state[1]),
                         self.l1 * np.sin(self.state[0]) + self.l2 * np.sin(self.state[1])])

    def update_ik_targ_angles(self, pos):
        pos *= np.maximum(np.abs(self.l1-self.l2)+1e-3, np.minimum(np.linalg.norm(pos), self.l1+self.l2-1e-3)) / np.linalg.norm(pos) # limit target to maximal stretch

        s1 = np.sqrt(2.*(self.l1**2.+self.l2**2.)*np.linalg.norm(pos)**2.-(self.l1**2.-self.l2**2.)**2.-np.linalg.norm(pos)**4.)
        s2 = np.sqrt((np.linalg.norm(pos)**2.-(self.l1-self.l2)**2.)*((self.l1+self.l2)**2.-np.linalg.norm(pos)**2.))

        th1 = 2.*np.arctan((2.*self.l1*pos[1]+s1)/((self.l1+pos[0])**2.+pos[1]**2.-self.l2**2.))
        th1 = [th1 + np.pi*2., th1 - np.pi*2., th1][np.argmin([np.abs(th1 + np.pi*2. - self.state[0]),
                                                              np.abs(th1 - np.pi*2. - self.state[0]),
                                                              np.abs(th1 - self.state[0])])]
        th2 = th1 - 2.*np.arctan(s2/(np.linalg.norm(pos)**2.-(self.l1-self.l2)**2.))
        th2 = [th2 + np.pi*2., th2 - np.pi*2., th2][np.argmin([np.abs(th2 + np.pi*2. - self.state[1]),
                                                              np.abs(th2 - np.pi*2. - self.state[1]),
                                                              np.abs(th2 - self.state[1])])]
        self.atarg[:] = np.array([th1, th2])
