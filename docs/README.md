# Release verification

`release_validation.json` records the comparison between the archived calculation source and the publication copy. All 50 original Python files have identical executable ASTs after excluding docstrings; changes are comments and technical documentation. New repository launch and prior-documentation utilities are listed separately in the root README.

The IDF publication copies differ only in comments. The topology/prior reconstruction reproduces the original reference JSON byte-for-byte, and reconstructing the processed E+ inputs reproduces all 18 arrays exactly. The 17-resistance, 8-state network diagram was rendered for inspection. Nine repository-launcher tests and nine archive/handoff-order tests passed. No new full MCMC or EnergyPlus run was performed for this release.

The source run used Python 3.12.10. Installing the pinned runtime is required before a new calculation. `python verify_repository.py` verifies the distributed bytes without installing numerical dependencies; `python run.py verify` checks the workflow manifest. Fresh computation uses the root launcher and creates new local files under `runs/`.

`01_energyplus/provenance.json` records input provenance and the limits of executed-IDF/EPW identity evidence. `04_results/provenance.json` identifies the completed run and the source result archives used for the summary tables. Comments, documentation and local machine paths were prepared for publication; scientific values and convergence thresholds were retained.
