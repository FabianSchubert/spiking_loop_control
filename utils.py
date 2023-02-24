#! /usr/bin/env python3

from pygenn.genn_model import (create_custom_init_var_snippet_class,
                               create_custom_neuron_class,
                               create_dpf_class)

from dataclasses import dataclass, field
import typing

from itertools import cycle

import numpy as np

norm_w_no_autapse_model = create_custom_init_var_snippet_class(
        "norm_w_no_autapse",
        param_names=["mean", "sd"],
        var_init_code="""
            $(value) = ($(id_pre) == $(id_post)) ? 0.0 : ($(mean) + $(gennrand_normal) * $(sd));
        """
    )

@dataclass
class EulerMaruyama:

    f: typing.Callable
    C: np.ndarray
    dt: float
    size: int
    buffer_rand_samples: typing.Union[bool, int] = False

    def __post_init__(self):
        self.dtsqrt = np.sqrt(self.dt)

        U, S, V = np.linalg.svd(self.C)

        self.P_SIGM = U @ np.diag(np.sqrt(S))

        if self.buffer_rand_samples:
            self.rnd_dat = np.random.normal(0.,1.,(self.buffer_rand_samples, self.size)) @ self.P_SIGM
            self.cycleid = cycle(range(self.buffer_rand_samples))
            self.psi = lambda : self.rnd_dat[self.cycleid.__next__()]
        else:
            self.psi = lambda: self.P_SIGM @ np.random.normal(0.,1.,(self.size))


    def step(self, t, x):
        return x + self.dt * self.f(t, x) + self.dtsqrt * self.psi()
