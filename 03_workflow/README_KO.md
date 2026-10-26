# Calibration and reduction workflow

This directory contains the executable single-setpoint pipeline. The filename is retained for the archive controller; documentation is in English.

Use the repository-root `run.py` interface. It provides `verify`, `start`, `resume`, `status`, `stop` and `export`, and creates fresh working files under `runs/` by default. The completed-study summaries are stored separately in `04_results/`.

## Execution order

1. Verify the source and input checksums.
2. Run the 17R8C joint calibration with four independent chains.
3. Write and CRC-verify `Initial_MCMC_results.zip` before reduction.
4. Require the actual retained-sample convergence and numerical-health gates to pass.
5. Transfer one coherent representative draw into the reduction model and verify temperature-trajectory parity.
6. Enumerate all current component pairs, fit each merged component, and screen convergence and validation RMSE.
7. Select a Pareto-front candidate using d_R and d_C; repeat from the adopted child.
8. Perform binary and numeric subgroup discovery; export reduction/rule and combined archives.

An initial run that fails its gate is archived and stops before reduction. A successful process exit alone is not a convergence criterion. Each checkout maintains one saved run; resume uses only its own checkpoints.

## Input files

`initial/inputs/HIBO_MCMC_1min.csv` is the complete July EnergyPlus output used for this calculation. `data.npz` contains the runtime arrays. `build_data.py` reconstructs those arrays and checks timestamps, calendar, thermostat control, physical inputs and data partitions. It writes derived files in its own directory; use a copied input directory for reconstruction checks so the frozen package manifest remains valid.

The matching IDF is `initial/inputs/HIBO.idf`. The model, weather, network and prior derivation are documented in the repository's first two numbered directories. Historical source names in `SOURCE_FILES.json` identify provenance; those earlier archives are not runtime dependencies.

## Statistical settings

The 17R8C initial model estimates 17 R, 8 C, 28 independent allocation coordinates and two noise parameters. Its 32 reported allocation fractions conserve each source across eight states. Inputs are one-minute values, thermal integration uses two 30-second backward-Euler steps, and the AR(1) likelihood uses five-minute endpoints. July 1 conditions the states, July 2–22 is training, and July 23–31 is validation/model selection. States remain continuous across splits.

`initial/CASE5_SETTINGS.json` records the fixed sampler configuration. The legacy filename is part of the runtime interface. Adaptive warmup is 12,000 iterations; the target HMC acceptance is 0.95. Frozen settling is evaluated at 8,000 and subsequent 4,000-iteration intervals, with two consecutive nonoverlapping confirmations and a pooled check. Its earliest release is 12,000 iterations. Retained production requires at least 5,000 draws per chain and both physical and ordered R-hat < 1.10, bulk ESS >= 400 and tail ESS >= 400. Settling ESS is descriptive, not the production ESS gate.

The initial sampler combines HMC, an exterior-tempered proposal, conditional heat updates, residue updates and an auxiliary exterior/air transport. It is a custom implementation; see `initial/SAMPLER_DESIGN.md` for its transformations, acceptance rules and limitations.

The reduction uses four-chain adaptive random-walk Metropolis. Only new merged R/C and four allocation coordinates are fitted; nonmerged blocks, air capacity and noise remain fixed. Air allocation follows conservation. Massless-only pairs have direct R coordinates and no C or allocation coordinates. All pairs are allowed regardless of adjacency or window host. A single PASS can be selected. See `reduction/SCIENTIFIC_METHOD.md` for the operators, Pareto objectives and subgroup definition.

## Integrity

`MANIFEST_SHA256.json` covers this workflow package. The initial stage also has a separate source/input manifest. Editing any tracked input after a run starts invalidates that run's resume identity. Comments and documentation were prepared for publication without changing executable mathematical statements. The original completed-run source hashes remain historical provenance rather than checksums of the publication copy.
