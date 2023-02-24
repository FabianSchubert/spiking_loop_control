#! /usr/bin/env python3

from pygenn.genn_model import (create_custom_neuron_class,
                               create_dpf_class)

import numpy as np

class External:

    model = create_custom_neuron_class(
        "external_neuron",
        sim_code="""
            $(u) = $(Isyn);
        """,
        param_names=[],
        var_name_types=[("x", "scalar"), ("u", "scalar")]
    )

class Noise:

    params = {}

    var_init = {"psi": 0.0}

    model = create_custom_neuron_class(
        "noise_neuron",
        sim_code="""
            $(psi) = $(gennrand_normal);
        """,
        param_names=[],
        var_name_types=[("psi", "scalar")]
    )

class Lif:

    params = {
        "l": 1.0,
        "vt": 1.0,
        "vr": 0.0
    }

    var_init = {
        "v": 0.0,
        "spike": 0
    }

    model = create_custom_neuron_class(
            "lif_slow_fast",
            sim_code="""
                $(v) += (DT * ($(Isyn_fast) + $(Isyn_slow)
                             + $(Isyn_ds) + $(Isyn_z)
                             - $(l)*$(v))
                             + $(DTSQRT) * $(Isyn_noise)
                        );

                $(spike) = 0;
                if($(v) > $(vt)){
                    $(spike) = 1;
                    $(v) = $(vr);
                }

                // reset spikeCount
                if($(id) == 0){
                    *($(spikeCount)) = 0;
                }
            """,
            threshold_condition_code="""
                //my threshold code
                ($(spike)
                && (*($(spikeCount)) == 0)
                && (atomicCAS($(spikeCount), 0, 1) == 0))
            """,
            reset_code="""
                //$(v) = $(vr);
            """,
            param_names=["l", "vt", "vr"],
            var_name_types=[("v", "scalar"),
                            ("spike", "int")],
            additional_input_vars=[("Isyn_fast", "scalar", 0.0),
                                   ("Isyn_slow", "scalar", 0,0),
                                   ("Isyn_noise", "scalar", 0.0),
                                   ("Isyn_ds", "scalar", 0.0),
                                   ("Isyn_z", "scalar", 0.0)],
            derived_params=[("DTSQRT", 
                create_dpf_class(lambda pars, dt: np.sqrt(dt))())],
            extra_global_params=[("spikeCount", "int*")],
            is_auto_refractory_required=False
        )

