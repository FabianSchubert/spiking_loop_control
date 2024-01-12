import numpy as np


class Cartpole:
    def __init__(self, l, m1, m2, damp_x, damp_theta, g, dt):
        self.l = l
        self.m1 = m1
        self.m2 = m2
        self.damp_x = damp_x
        self.damp_theta = damp_theta
        self.g = g
        self.dt = dt

        self.state = np.zeros((4))
        self._targ = np.zeros((2))

    @property
    def targ(self):
        return np.array([self._targ[0], self._targ[1], 0.0, 0.0])

    @property
    def A(self):
        a = np.zeros((4, 4))
        a[0, 2] = 1.0
        a[1, 3] = 1.0
        a[2, 1] = self.m2 * self.g / self.m1
        a[2, 2] = -self.damp_x / self.m1
        a[2, 3] = -self.damp_theta / (self.l * self.m1)
        a[3, 1] = (self.m1 + self.m2) * self.g / (self.l * self.m1)
        a[3, 2] = -self.damp_x / (self.l * self.m1)
        a[3, 3] = (
            -self.damp_theta * (1.0 + self.m1 / self.m2) / (self.l**2.0 * self.m1)
        )

        return a

    @property
    def B(self):
        b = np.zeros((4, 1))
        b[2, 0] = 1.0 / self.m1
        b[3, 0] = 1.0 / (self.l * self.m1)

        return b

    @property
    def base_pos(self):
        return np.array([self.state[0], 0.0])

    @property
    def tip_pos(self):
        return np.array(
            [
                self.state[0] - self.l * np.sin(self.state[1]),
                self.l * np.cos(self.state[1]),
            ]
        )

    def step(self, action):
        vels = self.state[2:]

        x = self.state[0]
        theta = self.state[1]
        vx = vels[0]
        vtheta = vels[1]

        fx = action[0] - self.damp_x * vx
        ftheta = -self.damp_theta * vtheta

        # """
        dvxdt = (
            -self.l * self.m2 * np.sin(theta) * vtheta**2.0
            + self.m2 * self.g * np.cos(theta) * np.sin(theta)
            + ftheta * np.cos(theta) / self.l
            + fx
        ) / (self.m1 + self.m2 * (1.0 - np.cos(theta) ** 2.0))

        dvthetadt = (
            -self.l * self.m2 * np.cos(theta) * np.sin(theta) * vtheta**2.0
            + self.g * (self.m1 + self.m2) * np.sin(theta)
            + fx * np.cos(theta)
            + ftheta * (1.0 + self.m1 / self.m2) / self.l
        ) / (self.l * (self.m1 + self.m2 * (1.0 - np.cos(theta) ** 2.0)))
        # """

        """
        dvxdt = (self.m2 * self.g * theta + fx + ftheta / self.l) / self.m1
        dvthetadt = (
            self.g * (self.m1 + self.m2) * theta
            + fx
            + ftheta * (1.0 + self.m1 / self.m2) / self.l
        ) / (self.l * self.m1)
        # """

        self.state[0] += self.dt * vx
        self.state[1] += self.dt * vtheta
        self.state[2] += self.dt * dvxdt
        self.state[3] += self.dt * dvthetadt
