#! /usr/bin/env python3

from pygenn.genn_model import create_custom_weight_update_class

fast_syn = {
	"matrix_type": "DENSE_INDIVIDUALG",
	"delay_steps": 0,
	"w_update_model": "StaticPulse",
	"wu_param_space": {},
	#"wu_var_space": set when adding a synapse population
	"wu_pre_var_space": {},
    "wu_post_var_space": {},
    "postsyn_model": "DeltaCurr",
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
    "postsyn_model": "ExpCurr",
    "ps_param_space": {"tau": 0.1},
    "ps_var_space": {}
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








