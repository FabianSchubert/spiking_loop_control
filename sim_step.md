## Simulation Step in GeNN

    stepTime()

-     updateSynapses()

	+     for all presynaptic spikes detected/trigerred in a presynaptic cell
		  in the previous simulation step:

		*     Run weight update code, e.g. addToInSyn.

-     updateNeurons()

	+     Isyn = 0
          <add. inp. var.> = <your init. value>
	+     Run code defined in <apply_input_code> in
          your postsynaptic model. The code should
          set Isyn (or your additional input variable).
          In any case, the code should be written using
          Isyn as a template input variable.
	+     Run <decay_code> from the postsynaptic model.
          This is where you should modify inSyn (NOT Isyn,
          which is set to zero in every call of
          updateNeurons()), e.g. to implement some form
          of exponential decay of the input current.
	+ Note: If you don't set inSyn to zero in `<apply_input_code>` or `<decay_code>`,
       it will remain at its current value, and `updateSynapse()` will use this value as a
       starting point in the next simulation step.
	+     Run <sim_code> of your neuron model.
	+     if( <threshold_condition_code> ):
		*     Run <reset code>