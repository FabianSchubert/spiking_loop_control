#! /usr/bin/env python3

from pygenn.genn_model import (create_custom_weight_update_class,
                               create_custom_postsynaptic_class,
                               create_dpf_class)

import numpy as np

ps_delta_curr_norm = create_custom_postsynaptic_class(
    "ps_delta_norm",
    param_names=[],
    var_name_types=[],
    decay_code=None,
    apply_input_code="""
        $(Isyn) += $(inSyn);
        $(inSyn) = 0.;
    """
    )

ps_r_curr = create_custom_postsynaptic_class(
    "ps_r_curr",
    param_names=["l"],
    derived_params=[("decFact",
                create_dpf_class(lambda pars, dt: np.exp(-dt*pars[0]))())],
    var_name_types=[("inSynCustom", "scalar")],
    apply_input_code="""
        //apply input code (comes before decay code)
        $(inSynCustom) += $(inSyn);
        $(inSyn) = 0.;
        $(Isyn) += $(inSynCustom);
    """,
    decay_code="""
        //decay code
        $(inSynCustom) *= $(decFact);
    """
    )

fast_syn = {
    "matrix_type": "DENSE_INDIVIDUALG",
    "delay_steps": 0,
    "w_update_model": "StaticPulse",
    "wu_param_space": {},
    #"wu_var_space": set when adding a synapse population
    "wu_pre_var_space": {},
    "wu_post_var_space": {},
    "postsyn_model": ps_delta_curr_norm,
    "ps_param_space": {},
    "ps_var_space": {}
}

slow_syn = {
    "matrix_type": "DENSE_INDIVIDUALG",
    "delay_steps": 0,
    "w_update_model": "StaticPulse",
    "wu_param_space": {},
    #"wu_var_space": set when adding a synapse population
    "wu_pre_var_space": {},
    "wu_post_var_space": {},
    "postsyn_model": ps_r_curr,
    #"ps_param_space": {"l": 1.0},
    #"ps_var_space": {"inSynCustom": 0.0}
}

continuous_external_syn = {
    "matrix_type": "DENSE_INDIVIDUALG",
    "delay_steps": 0,
    "w_update_model": create_custom_weight_update_class(
        class_name="wu_mod_observe",
        var_name_types=[("g", "scalar")],
        param_names=[],
        synapse_dynamics_code="$(addToInSyn, $(g) * $(x_pre));"
    ),
    "wu_param_space": {},
    #"wu_var_space": set when adding a synapse population
    "wu_pre_var_space": {},
    "wu_post_var_space": {},
    "postsyn_model": "DeltaCurr",
    "ps_param_space": {},
    "ps_var_space": {}
}

noise_syn = {
    "matrix_type": "DENSE_INDIVIDUALG",
    "delay_steps": 0,
    "w_update_model": create_custom_weight_update_class(
        class_name="wu_mod_noise",
        var_name_types=[("g", "scalar")],
        param_names=[],
        synapse_dynamics_code="$(addToInSyn, $(g) * $(psi_pre));"
    ),
    "wu_param_space": {},
    #"wu_var_space": set when adding a synapse population
    "wu_pre_var_space": {},
    "wu_post_var_space": {},
    "postsyn_model": "DeltaCurr",
    "ps_param_space": {},
    "ps_var_space": {}
}

