#! /usr/bin/env python3

from pygenn.genn_model import (create_custom_neuron_class,
                               create_dpf_class,
                               create_custom_custom_update_class)

import numpy as np

def damped_spring_mass(x, u, k, g):
    return np.array([x[1],
                    -x[0]*k -x[1]*g + u[1]])

def lin_system(x, u_eff, A):
    return A@x + u_eff