# Numerical-stability rationale

The prior single-setpoint run did not release retained production samples because its frozen-settling numerical-health gate repeatedly failed. Small R-hat values alone did not establish a usable computation. Invalid HMC proposals and large integration-energy errors occurred especially in slow, high-capacity floor states.

The numerical review checked target gradients, dense-metric boundary reflections and paired-momentum step-size probes. At the examined states, reducing the HMC integration step removed the numerical errors seen with the previous step. These checks motivated the version-3 target acceptance of 0.95 and 12,000 adaptive warmup iterations. They did not introduce a floor-specific prior or a bound chosen from the failed draws.

The HMC trajectory integration step is distinct from the thermal model's 30-second substep. The physical model, minute forcing, five-minute likelihood, priors and convergence thresholds were retained. Auxiliary exterior/air proposals remain custom components of the initial sampler; the reduction uses a separate conditional random-walk sampler.

The completed-run summary and its uncertainty/validation limits are in the repository README and `04_results/`. The retained physical and ordered diagnostics pass their prescribed criteria. This demonstrates the implemented numerical checks for that calculation; it does not establish unique recovery of all thermal parameters or precise probabilities for rarely visited posterior regions.
