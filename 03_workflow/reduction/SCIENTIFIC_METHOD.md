# HIBO Lab B reduction v3: all pairs and merged R/C/alpha calibration

This pipeline first estimates a fresh 17R8C initial model from the supplied single-setpoint July CSV. After saving and verifying the initial-result ZIP and passing convergence gates, it supplies one new coherent joint draw to the unchanged RCA v3 numerical reduction method. Fresh calculations do not import previous Case 5 estimates or completed candidate records. Published completed-run summaries are separate from runtime inputs. The parent README documents the integrated execution and archive paths.

## Workflow and candidate policy

Freeze M_k, enumerate every unordered envelope-block pair, initialize and independently fit every candidate, archive the complete wave, compare PASS candidates, and adopt one child before repeating. Zone air remains a permanent state outside the candidate list. For n components there are n(n−1)/2 pairs. Initial n=10 gives **45 pairs**: 21 massive–massive, 21 massive–window and 3 window–window.

Physical adjacency and original host mappings are provenance only. They do not prune candidates or invalidate a merge. Non-touching surfaces and windows attached to other walls may merge. All-pairs generation at visited parents is not exhaustive enumeration of every reduction path or proof of a global minimum.

Distinct boundary-temperature inputs—plenum, outdoor, neighboring Space 102–Plus and ground—remain explicit. A massive block has one R per active boundary, one air-side R and one C. A massless block has one direct boundary-to-air R per active boundary and no C. Actual whole-model counts are:

\[
N_R=\sum_{i\in\mathrm{massive}}(|\mathcal B_i|+1)+\sum_{w\in\mathrm{massless}}|\mathcal B_w|,
\qquad N_C=N_{\mathrm{massive}}+1.
\]

The added C is zone air. Labels count actual elements rather than sampled coordinates or components. Different-boundary window grouping and generalized nonhost window absorption that adds a new boundary port can reduce component count while leaving N_R and N_C unchanged; such candidates are still evaluated and recorded with `dimension_reduced=false`. No artificial C is introduced.

## Thermal equations and initializers

For massive block i:

\[
C_i\dot T_i=\sum_{b\in\mathcal B_i}(T_b-T_i)/R_{i,b}+(T_z-T_i)/R_{i,z}+\sum_s\alpha_{s,i}Q_s.
\]

Zone air receives heat from the massive blocks, direct massless branches, its allocated gains, HVAC supply and infiltration. HVAC exchange is g_HVAC(T_terminal−T_z), where terminal temperature is downstream of reheat and T_z is the simulated indoor temperature. Infiltration uses g_inf(T_outdoor−T_z). Input derivations are retained from Case 5.

### Massive–massive

Initialize C_m=C_A+C_B and alpha_m=alpha_A+alpha_B. Sum conductances separately at each boundary and at the air-side port; an absent boundary contributes zero conductance, not zero R. Apply this operator separately to the IDF reference values. This merges two temperature states and is generally not transiently equivalent. **Initialized merged R, C and alpha are subsequently fitted; the sums are not hard fitting constraints.**

### Massless–massive

Any massive block may absorb any massless block. For massive air-side conductance G_z, boundary conductances G_b and direct massless conductances W_b, define:

\[
f=G_z/(G_z+\sum_bG_b),\quad K'_b=fG_b+W_b,
\quad G'_b=K'_b/f,\quad G'_z=\sum_bK'_b/(1-f).
\]

Remove the absorbed direct branches exactly once. Initialize C and alpha from the massive constituent. The combined steady-state boundary-to-air conductance coefficients are preserved at initialization, including newly introduced boundary ports. This is not an exact transient contraction or a constraint during fitting. Transform the IDF reference network independently with the same operator.

### Massless–massless

Sum direct conductances separately by boundary, preserving different boundary inputs and paths. The merged block remains C=0 and alpha=0, with no temperature state. Re-estimate its active direct R values. C/alpha coordinates are absent because these quantities do not exist in the massless network.

## Fitted and fixed parameters

A massive merged block has all active boundary R, its air-side R, its C and four source allocation fractions fitted. Other blocks' R/C and alpha, zone-air C and hourly-reference residual sigma/rho remain fixed. **Zone-air alpha is a dependent remainder**, not a fixed quantity. An all-massless merged block has only its active direct R values fitted.

Sources are solar, lighting, occupant sensible and equipment sensible heat. For each source s, define a fixed local allocation budget from the initialized child and parent air:

\[
B_s=\alpha^{initial}_{s,m}+\alpha^{parent}_{s,z},\qquad
\alpha_{s,m}=B_sp_s,\qquad\alpha_{s,z}=B_s(1-p_s).
\]

All other solid allocations remain unchanged, preserving nonnegativity and a source-wise sum of one. Air alpha is not an extra independently sampled parameter. Let a be the number of original massive components represented by the merged block. Use independent p_s~Beta(a,1). This is consistent with aggregation and conditioning of an original Dirichlet(1,…,1) over the seven original massive states and air. Windows do not contribute a state or increase a. The budget and other allocations are fixed fitted-parent values; this is conditional plug-in calibration, not propagation of the previous joint posterior.

### Priors and transformations

For positive R/C parameter x and separately propagated IDF reference mean m:

\[
\ln x\sim N(\ln m-\tfrac12\ln5,\;\mathrm{variance}=\ln5).
\]

Physical mean is m and SD is 2m. The aggregate lognormal prior is a new modeling choice, not the exact induced distribution of sums/parallel combinations of previous random values.

Coordinates are ln x for R/C and z_s=logit(p_s) for allocation. A Beta(a,b) prior on p has transformed log density **including the Jacobian**:

\[
\log\pi(z)=a\log\operatorname{sigmoid}(z)+b\log\operatorname{sigmoid}(-z)+constant.
\]

Here b=1. No Gaussian R/C prior is applied to alpha coordinates. Stable sigmoid/log-sigmoid evaluation avoids replacing the intended distribution by hard clipping. Saved coordinates and physical values are explicitly distinguished.

## Data, solver and likelihood

| Partition | Dates | One-minute rows | Use |
|---|---|---:|---|
| Conditioning | July 1 | 1,440 | State evolution; excluded from fit and errors |
| Training | July 2–22 | 30,240 | 6,048 five-minute likelihood endpoints |
| Selection-validation | July 23–31 | 12,960 | Minute RMSE gate and selection |

The 44,640-minute trajectory is continuous without partition-boundary state resets. All states start at the first E+ indoor temperature. One-minute forcing is held constant during two 30-second backward-Euler substeps. Likelihood uses each fifth minute endpoint, not five-minute means or only setpoint-change events.

The hourly-reference residual parameters fixed to the NEW initial joint draw map to five minutes:

\[
\phi_5=\rho_h^{1/12},\qquad\sigma_5=\sigma_h\sqrt{(1-\phi_5^2)/(1-\rho_h^2)}.
\]

For e_j=T_model,j−T_E+,j, the stationary AR(1) likelihood uses (1−phi_5²)e_0² for the first residual and (e_j−phi_5e_(j−1))² thereafter, including the sigma and stationary normalization terms. Validation observations and future forcing do not enter the training likelihood. Fixed noise does not establish equal residual distributions across topologies.

M₀ uses the highest original labeled log-posterior joint draw among this new run's retained samples. Its temperature trajectory and validation RMSE are regenerated and checked against the initial export during `verify`. No numerical result is preassigned.

## Sampling, diagnostics and representative vector

Four independent chains use 70% full-block Gaussian and 30% single-coordinate Gaussian Metropolis–Hastings proposals in unconstrained coordinates. During warmup covariance and full-block/per-coordinate scales adapt separately. Acceptance targets are 0.234 for block proposals and 0.44 for coordinate proposals. Settling and retained production use frozen proposals. This is a reduced-block sampler, not the topology-specific Case 5 kernel.

| Setting per chain | Primary | Predeclared rescue |
|---|---:|---:|
| Warmup | 12,000 | 24,000 |
| Frozen settling, discarded | 12,000 | 12,000 |
| Minimum retained draws | 6,000 | 6,000 |
| Maximum retained draws | 24,000 | 48,000 |
| Diagnostic interval | 1,000 | 1,000 |

All retained production draws are used without thinning or posthoc trimming to obtain a pass. Full RNG/adaptation states are checkpointed. Compatible resume preserves the sequence and incompatible code/config/input/runtime identities are rejected. Atomic writes use bounded retry for temporary permission locks; Windows results default outside OneDrive.

Every free parameter must meet rank-normalized split/folded R-hat<1.10, bulk ESS≥400 and tail ESS≥400 in both unconstrained and physical representations. Tail ESS is the minimum of explicit 5% and 95% quantile ESS. Alpha's physical transform is its bounded allocation, not exp(logit). These empirical criteria do not prove exact convergence or physical identifiability.

The representative is the retained **joint** draw with highest target density in the unconstrained coordinate measure. It is not a vector of separately chosen marginal medians or a guaranteed global MAP. Density maximizers depend on coordinates, and unnormalized targets across candidate models are not model evidence.

## Gates, rescue and comparative adoption

- PASS: convergence plus representative minute-resolution validation RMSE≤1°C.
- FAIL: converged but RMSE>1°C.
- UNRESOLVED: budget exhausted without all convergence criteria, regardless of RMSE.
- Smoke records are never eligible for adoption.

Complete the full primary wave before selection. **One PASS is sufficient for adoption.** If no PASS exists and unresolved alternatives remain, freeze the full unresolved list and execute the predeclared fresh rescue attempt for all of them. Once scheduled, finish the rescue list even if an early attempt produces a PASS. With one or more primary PASS candidates, do not automatically rescue remaining unresolved alternatives.

Rescue uses the same frozen parent/target and independent seeds. Attempts are not pooled; the later scheduled attempt is authoritative even if worse. Rule datasets have one final outcome per trial.

After evaluation: zero PASS preserves the parent and stops (explicitly distinguishing unresolved alternatives); one PASS adopts that candidate; multiple PASS use the Pareto rule below. A final singleton pair is eligible. Stop normally when only one envelope block remains, since no pair exists. Zero PASS is a result under this evaluation budget, not a proof of physical impossibility.

## Postfit distances and Pareto

Compare the fitted merged block in M_(k+1) with each constituent in frozen M_k. Do not substitute premerge pair similarity for this postfit comparison.

For a massive block, R_eff=R_air+(sum_b 1/R_b)^(-1). For a massless block, R_eff=(sum_b 1/R_direct,b)^(-1). These scalar ranking descriptors do not collapse the simulation's distinct boundary inputs.

\[
d_R=\sqrt{\tfrac12\sum_{i\in\{A,B\}}[\ln(R_{eff,m}/R_{eff,i})]^2},
\qquad d_C=\sqrt{|I_C|^{-1}\sum_{i\in I_C}[\ln(C_m/C_i)]^2}.
\]

I_C includes massive constituents only. For massless–massive, compare C with the massive constituent alone. For massless–massless, no C exists before or after: d_C=0 is an explicit no-capacitance-change convention and C_comparison_count=0. Do not compute log(0) or invent a small C. This convention can favor massless grouping on the C objective and must remain visible in interpretation.

PASS candidates enter Pareto fronts minimizing d_R and d_C. A dominates B if no worse in both and strictly better in at least one. Choose from front 1 by validation RMSE, child C count, child R count and stable trial ID. A singleton PASS receives front 1 and is adopted. The selected path records whether there were genuinely multiple PASS alternatives.

No absolute d_R/d_C upper limits are introduced. The distances combine structural aggregation and refitting: merging two equal C values without refitting already gives d_C=ln2. They are not pure refit penalties, conservation constraints or absolute physical loss. Portwise R changes, fitted C versus parent C sum, and initializer-to-fit changes are supplementary and do not rank candidates. Ranking uses a representative vector, not a posterior probability of dominance.

## Binary subgroup discovery (retained alongside numeric analysis)

Use unique final resolved PASS/FAIL records. UNRESOLVED/stopped records are excluded from outcome rates and reported separately. Predictors are prefit pair distances, constituent effective R/C, window/operator indicators, boundary counts and parent/child element counts. Fitted values, postfit ranking distances, RMSE, diagnostic results and adoption decisions are excluded from predictors.

Beam search uses up to two conditions, beam width 20, at least 20 resolved trials and support of at least five resolved trials. Unweighted WRAcc is:

\[
WRAcc=(n_{covered,resolved}/n_{resolved})\,[p_{target\mid covered,resolved}-p_{target\mid resolved}].
\]

This uses Lavrač et al. (2004, §2.2), not the full CN2-SD weighted-covering algorithm. If evidence or outcome contrast is insufficient, no rule is inferred and the reason is reported. Running rule discovery does not guarantee finding a meaningful rule.

Trials share one building, forcing, sequential parents and repeatedly used selection-validation period. They are not independent building replications. Rules describe this HIBO July evaluated path and are not universal, causal or externally validated laws. Temperature-bin and transition diagnostics are descriptive; v3 does not add unannounced acceptance thresholds based on them.

## Numeric subgroup discovery (Steps 12–13)

Run independently of the binary outcome-contrast check. Default population is converged PASS massive–massive candidates. Exclude window pairs from this analysis so the structural d_C=0 convention does not manufacture a rule. Keep FAIL and UNRESOLVED labels unchanged. Records with duplicate trial IDs or invalid required inputs are rejected, not silently duplicated.

Use only prefit pair R/C distances, R/C minima/maxima and boundary counts as selectors. Do not select on stage, postfit R/C, d_R/d_C, RMSE, diagnostics, Pareto rank or adoption. Postfit d_R/d_C are numeric targets. Compute average-tie rank percentiles `(rank - 0.5)/n_stage` separately within each parent stage; each stage has baseline 0.5.

Search short conjunctions (at most two selectors), support >=10, at least three parent stages and coverage <=85%. Thresholds derive from prefit descriptor quantiles and readable ratio thresholds. The exploratory quality score is `sqrt(n/N) * min(0.5 - mean_rank_R, 0.5 - mean_rank_C)`. Both improvements must be positive. This is an explicit analysis choice, not a uniquely established standard. A support-constrained search can return no qualifying rule; the code reports why rather than forcing a rule.

Separately report readable conditions R ratio <=3 and (R ratio <=4 AND C ratio <=4), recomputed against the actual records. They are labeled exploratory reference conditions, not hard-coded discoveries or universal cutoffs. Report unique member-set pair counts, covered stages, same-stage-weighted raw means, covered trial IDs and exceptions. `R_eff` means the fitted current parent descriptor, not original IDF material resistance.

Rules are postprocessing only and do not prune candidates, skip MCMC, override PASS gates or change Pareto selection. Same-pair repeats and overlapping rules are not independent replications. Similar prefit R/C partly imposes a mathematical lower bound on postfit log-distance; report excess-distance sensitivity rather than claiming a new physical law. No held-out rule validation or significance claim is made.

## Outputs and validation scope

Each candidate has preserved `attempt_01_primary` and optional `attempt_02_rescue` folders. Its root result identifies the authoritative attempt and artifact directory. Posterior files distinguish `coordinates`, `coordinate_transforms`, and `physical_parameters`. Summaries use physical R/C units and actual alpha fractions; derived air allocations are labeled dependent.

`all_trials.*` has one final record per candidate; `selected_path.json` is the adopted sequence; `selected_model.json` is the last accepted network; `rules/binary/` and `rules/numeric/` contain separate exploratory results or absence reasons; `rules/summary.json` reports both. The integrated controller supplies an explicit work path and saves named result ZIPs in the top-level `exports` folder.

Delivery validation details are in the parent `validation/V2_VALIDATION.json` (historical baseline checks are in `validation/baseline_v1/BUILD_VALIDATION.json`). Smoke tests check execution and M0 parity, not production convergence. No previous posterior or reduction result is shipped as a new result.

## References

1. Vehtari et al. (2021), *Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC*. https://arxiv.org/abs/1903.08008
2. ArviZ diagnostic APIs; tested versions are pinned in `requirements.txt`. https://python.arviz.org/
3. Stan Functions Reference, simplex distributions. https://mc-stan.org/docs/functions-reference/simplex_distributions.html
4. Lavrač et al. (2004), *Subgroup Discovery with CN2-SD*, JMLR 5:153–188. https://www.jmlr.org/papers/volume5/lavrac04a/lavrac04a.pdf
