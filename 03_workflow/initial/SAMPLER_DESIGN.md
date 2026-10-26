# v3 numerical-stability update

Target acceptance changes from 0.90 to 0.95 using the same dual-averaging algorithm. Adaptive warmup increases from 6,000 to 12,000, with metric updates at iterations 1,000, 2,000, 4,000 and 8,000. The final 4,000 iterations tune step sizes with a fixed metric; the auxiliary transport receives its final fit at warmup completion. All warmup is discarded, and every adapted setting freezes before settling. These changes affect computation, not the priors or data.

Production must also pass the inherited numerical-health screen for both all retained draws and the latest 4,000 draws. Each chain needs ordinary HMC attempts, at least one accepted HMC transition and finite health flags. At most 1% of ordinary HMC transitions may be numerically invalid or have absolute energy error above 1,000. These thresholds are inherited engineering guards, not universal literature defaults or posterior precision criteria. Export independently recomputes the screen from checkpoint statistics. R-hat and ESS remain separate requirements.

This is a custom HMC-based sampler, not an off-the-shelf NUTS implementation. The tempered bridge inherited from Case 5/v1 and the exterior/air transport introduced in v2 remain unchanged; no floor-specific proposal is added. See `docs/RESULT_REVIEW_KO.md` in the parent package for the source-backed audit and limitations.

# Single-setpoint v3 initial sampler

The scientific target remains the Case 5 17R8C model with all 55 independent parameters free: 17 R, 8 C, 28 gain-allocation coordinates and sigma/rho. All 32 allocation fractions are reported (59 reported values total). The July single-setpoint CSV is the archived input for this study. Fresh calculations do not import chains, posterior draws, fitted proposals or RNG state from an earlier study run. Resume uses only checkpoints from the same v3 run.

## Unchanged target and convergence requirements

IDF-based independent R/C priors, four Dirichlet allocation priors, the AR(1) noise prior and the likelihood are unchanged. Inputs remain one minute; backward-Euler integration uses two 30-second substeps; likelihood observations are five-minute endpoints. July 1 conditions the model, July 2–22 trains it, and July 23–31 supplies the validation/model-selection interval.

The sampler keeps four chains and seed 20261018. After the updated 12,000-iteration adaptive warmup, frozen settling is first assessed at iteration 8,000 and then every 4,000 iterations, up to 20,000. The existing requirement for two consecutive nonoverlapping passing windows plus pooled confirmation makes 12,000 the earliest successful settling completion. Settling does not require ESS >= 400. Retained production runs for at least 5,000 and at most 200,000 draws per chain. Physical and ordered production diagnostics must both satisfy R-hat < 1.10, bulk ESS >= 400 and tail ESS >= 400. No failed chain is removed.

The main mixture remains HMC 0.65 / exterior tempered bridge 0.25 / conditional heat 0.10. Noise and residue moves are unchanged. The v2 transport described below is an additional auxiliary kernel, not a model reduction or parameter constraint. RCA v3 reduction source and configuration remain unchanged from the archived implementation.

## Targeted exterior/air transport

The transport was introduced in v2 to address separation in the fastest ordered exterior response and its allocation/capacity relationship with zone air. Every fifth transition, when a proposal is available, this kernel jointly proposes:

- fastest ordered exterior log time constant, `q18`;
- its log effective-capacity / air-capacity partition, `q3`;
- log air capacity, `q24-logsumexp(0,q1,q3,q9,q13)`.

This chart replaces `(q18,q3,q24)` with unit Jacobian while other thermal coordinates remain fixed for this individual move. HMC still updates all 55 coordinates. The fastest ordered component is not a permanently named physical wall; exact conditional labeling and the original physical priors are preserved.

Two Gaussian proposal components are fitted from this run's discarded adaptive warmup, separated descriptively at `log(tau)=5`. The split is not a parameter bound, new prior or physical classification. Each component needs 64 observed positions. The latest 4,000 warmup iterations are used with covariance shrinkage 0.20. Fitting occurs at metric updates 1,000, 2,000, 4,000 and 8,000, and at warmup completion at 12,000. If a fit lacks data, an existing proposal from this same warmup is retained; if none exists, the kernel stays unavailable and this is logged. Gaussian proposal weights are 0.5/0.5, not estimated posterior mode probabilities.

The affine mixture transport chooses source/destination components and preserves the standardized residual. Same-component selections are identity moves and are counted separately in the helper. Invalid ordering is rejected once, never sorted, clipped or redrawn until valid. At a valid proposed thermal state, all allocation fractions are refreshed by the existing exact 512-particle multiple-try heat kernel. Its Metropolis correction includes normalized proposal densities, importance-weight sums, and selection after acceptance or rejection.

The three-coordinate transport operates in ordered-profile measure. The heat kernel operates in thermal log-gap measure and already includes the gap Jacobian in its target. The supplied proposal correction is therefore:

    log_g(s_old)-log_g(s_new) + sum(theta_old[GAPS]-theta_new[GAPS])

where `GAPS=[19,20,22]`. This cancels the coordinate-change factor once. No extra affine determinant is added: it is already accounted for by mixture reversibility. An invalid current importance bank raises an error; an invalid proposed bank is rejected. The Gaussian fit is a proposal device only, never a replacement prior or likelihood.

## Freeze, resume and records

All proposal means, covariances and weights freeze before discarded fixed-kernel settling. They are stored in `frozen_kernel.npz`, whose checksum is verified on resume. Warmup checkpoints include the current proposal and fit history, so interrupted adaptation resumes coherently. Production and settling do not refit the proposal. No previous-run proposal is shipped.

The log records attempted and accepted transport moves, actual region crossings, and per-chain occupancy of the descriptive fast region. These mobility records supplement R-hat and ESS; no acceptance rate alone establishes convergence. A successful fresh run is not guaranteed by the local engineering pilot. Pilot data are diagnostic evidence only and are not bundled as sampling input.

## Pipeline and result preservation

`Initial_MCMC_results.zip` is written and CRC-verified before any reduction handoff. The archive uses a writable file stream for Windows fsync. Missing required files or CRC failure prevent replacement and handoff. Nonconverged initial runs still export diagnostic checkpoints, then stop. On success, a coherent new joint posterior draw and its new noise values initialize the unchanged all-pairs RCA v3 reduction and numeric/binary subgroup analyses. The final topology is not forced to 5R2C.
