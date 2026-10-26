# Sampler revision history

Version 2 introduced an auxiliary exterior/air-capacity transport with exact multiple-try heat-allocation proposals. Proposal fitting uses only the new run's discarded adaptive warmup and is frozen before settling. The transport is attempted every fifth transition. Priors, physical topology and convergence thresholds were retained.

Version 3 raised the HMC target acceptance from 0.90 to 0.95 and adaptive warmup from 6,000 to 12,000 iterations. It retains the mathematical target and the auxiliary kernels, with a numerical-health screen evaluated from production checkpoints. The settings and production implementation in this repository are those used in the completed calculation.

The source comments and documentation were edited for publication. Historical run checkpoints are not portable resume inputs for this edited package. New calculations use the repository launcher.
