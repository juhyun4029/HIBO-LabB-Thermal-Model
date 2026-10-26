# Initial sampler: inherited Case 5 design

This package applies the Case 5 sampler to the NEW single-setpoint CSV. Historical development observations below describe the original sampler design, not results from this new run.

# Case 5 sampler design

Case 5 is a fresh Wide-data sampling experiment following the Case 3/4 audit. Old runs provide engineering evidence only. Their positions, posterior draws, initialization states, fitted proposals and RNG states are not imported.

## Fixed scientific target

The IDF-derived topology remains 17R8C. All 17 resistances, 8 capacities, 28 independent gain-allocation coordinates and two residual coordinates remain free (55 unknowns; 59 reported marginals when all 32 allocation fractions are shown). Original priors, the temperature-only likelihood, one-minute input data and two 30-second backward-Euler substeps are unchanged. Likelihood observations remain five-minute endpoints. July 1 conditions the state; July 2–22 trains; July 23–31 validates without resetting states.

The model's exact same-boundary permutation handling and conditional physical labeling are retained. Ordered exterior time-constant coordinates are not fixed physical wall labels. Priors and transformed-target/Jacobian corrections are not replaced by empirical fitting distributions.

## Proposal change

The Case 3/4 auxiliary thermal proposals almost never accepted, and their blocks omitted the ordered exterior time-constant coordinates implicated in the slow between-chain transitions. Case 5 replaces those thermal selections with a rank-one tempered-HMC exterior bridge. It heats only the momentum projection associated with fastest ordered exterior coordinate 18. Dense-metric Hamiltonian velocities couple all 55 coordinates, allowing R, C, allocation and noise to move jointly. The symmetric heating/cooling schedule returns to a cold endpoint and uses exact original-target Hamiltonian MH acceptance. Intermediate heated states are never retained. This dimensionless auxiliary sampler temperature is unrelated to indoor-temperature setpoints. No Gaussian-mixture or empirical-density proposal is fitted. The full original target is used at acceptance; compensating movement neither fixes parameters nor changes priors.

The main-kernel selection is fixed at HMC 0.65, exterior bridge 0.25 and heat-only refresh 0.10. These are main-kernel probabilities, not the fractions of all actual coordinate changes: noise and residue refreshes may also run within a transition. Bridge-specific acceptance and slow-coordinate movement are recorded separately, avoiding attribution of combined transition movement to one kernel.

Ordinary HMC retains target acceptance 0.90, requested path time Uniform(0.4,1.2), and maximum 1,024 leapfrog steps. The bridge step is 0.75 times the ordinary adapted step, frozen at the end of warmup; its peak temperature is log-uniform from 4 to 16. Tempering MH acceptance includes physical work from the heating/cooling protocol. It therefore does not drive a separate dual-averaging acceptance target: chasing an unattainable work-acceptance rate could shrink steps without improving exploration. Case 3/4 had approximately 89–90% HMC acceptance and no observed cap hits; raising that cap alone was not supported by their records. Heat refresh particles remain 512 and residue substeps remain 8. Numerical settings are specified in `CASE5_SETTINGS.json` and saved with the run.

## Prespecified phases

1. Four fresh independently initialized streams use seed 20261018. Fresh local optimization, independent dispersion and provenance checks remain part of initialization. Optimizer success does not establish stationarity.
2. Adaptive warmup: 6,000 transitions per chain. Fresh warmup adapts the dense metric and ordinary HMC step using the 0.90 acceptance target. The bridge uses 0.75 times that step, without its own acceptance-driven adaptation. These values and all kernel settings are then frozen and hashed.
3. Frozen burn-in: first 4,000 transitions per chain, excluded from assessment windows and production by the design specified before execution.
4. Frozen assessments: nonoverlapping 4,000-transition windows. Require two consecutive windows and their pooled 8,000 transitions to pass the rank-Rhat/numerical-health readiness screen. First possible pass is at 12,000 total frozen transitions. Maximum is 20,000 frozen transitions. Every frozen transition is discarded; no last-window samples are relabeled as posterior. ESS is reported descriptively at this readiness stage; the final 400 threshold applies to production. Exhausting this budget exits without production.
5. Production: starts from retained draw zero after the readiness gate. No proposal adaptation, thinning or posthoc deletion. First check at 5,000 retained draws per chain, then every 500. Stop at the first complete pass of both physical and ordered diagnostics: rank Rhat strictly below 1.10, Bulk ESS at least 400, and minimum explicit 5%/95% quantile ESS at least 400. All retained production samples participate. The absolute production limit is 200,000 per chain.

Separating predeclared burn-in from later readiness windows addresses the Case 3 cumulative-gate transient without retrospectively selecting the better part of the existing run. It does not establish that Case 3 had converged, and it does not explain away Case 4's continuing separation. Neither readiness checks nor short local build tests guarantee posterior exploration.

## Limited build pilot

A short real-data engineering pilot evaluated 48 rank-one proposals: all 48 were numerically valid and 25 were accepted. Fixed peak-temperature arms accepted 11/12 at T=4, 7/12 at T=16, 5/12 at T=64 and 2/12 at T=256. The largest accepted coordinate-18 jump was approximately 1.824. These are local proposal checks, not posterior chains, an ESS comparison, proof of travel between separated regions, or a convergence demonstration. The T=4 and T=16 endpoint arms together accepted 18/24 proposals. The shipped 4–16 log-uniform range is a conservative engineering choice requiring evaluation in the fresh run; those two endpoint-arm counts are not a measured acceptance rate for the continuous temperature mixture. Full thermal differential-evolution and full 53-dimensional Gaussian heated proposals did not produce acceptable cross-region candidates in their short pilots and are not enabled in Case 5.

## Isolation, checkpoints and export

The controller uses a Case 5 identity, package/input/source hashes and exact runtime-version checks. It refuses foreign checkpoints and never overwrites an existing fresh result folder. Resume is explicit and restricted to an interrupted own Case 5 run, not automatic extension of a terminal failed/completed experiment. OS advisory locking prevents a second run or export in the same package folder and releases after process death without sending signals to a PID.

The foreground controller records console output and creates `Initial_MCMC_internal.zip`; the parent controller separately saves `exports/Initial_MCMC_results.zip` after a normal sampler return, graceful stop or caught sampler failure. The ZIP includes saved checkpoints and diagnostic-only readiness records even when no posterior exists. Posterior export failures are recorded, and original checkpoints remain available. An abrupt OS termination or a disk failure can prevent final ZIP creation; `EXPORT_RESULTS.bat` prepares the bundle later from saved records.

Temperature validation includes overall and temperature/setpoint/response-phase errors and coherent joint-draw predictions. Passing MCMC diagnostics supports sampling of this target, not unique recovery of literal EnergyPlus layer properties. Compensating combinations and broad marginals are reported as uncertainty rather than concealed by narrow priors or an RMSE-only success rule.
