"""CPU MCMC with staged warmup and a frozen production kernel.

The model has 17 resistance, 8 capacity, 28 independent allocation, and
2 noise parameters. Fresh runs initialize independent chains and random
streams. Sampling ends at the first complete physical and ordered-coordinate
diagnostic pass."""
import os
for _environment in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_environment] = '1'
import argparse
import hashlib
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from canonical_model import to_v
from global_transport import to_gap
from model import ROOT
from multiple_try_heat import transition as heat_transition
from reporting import diagnostics, canonical_diagnostics
from diagnostic_parallel import diagnostic_definition, runtime_environment
from slice_transition import coordinate_slice
from tuned_adaptation import (DualAveraging, momentum_metric,
                              pooled_covariance, warmup_windows)

DIMENSION = 55
CHAINS = 4
STATISTIC_NAMES = [
    'kind', 'accepted', 'acceptance_probability', 'importance_ess',
    'maximum_importance_weight', 'invalid_or_abs_energy_over_1000',
    'nonidentity', 'residue_attempts', 'residue_valid', 'residue_accepts',
    'residue_nontrivial', 'hmc_energy_error', 'hmc_leapfrog_steps',
    'hmc_step_size', 'hmc_integration_time', 'hmc_length_capped',
    'combined_transition_standardized_jump', 'transition_seconds',
    'main_thermal_diagonal_metric_sqjump', 'main_alpha_fraction_sqjump',
    'noise_coordinate_sqjump', 'residue_thermal_diagonal_metric_sqjump',
    'residue_alpha_fraction_sqjump', 'main_seconds', 'auxiliary_seconds',
    'bridge_nominal_peak_temperature', 'main_exterior_logtau_change',
    'main_exterior_region_crossing', 'main_numerically_invalid',
    'main_abs_energy_over_1000',
    'transport_attempted', 'transport_valid', 'transport_accepted',
    'transport_acceptance_probability', 'transport_cross_region_proposal',
    'transport_region_crossing', 'transport_seconds', 'final_exterior_region_fast']


def kernel_summary(rows):
    """Movement immediately after each main proposal, before auxiliary moves."""
    rows=np.asarray(rows)
    result=dict(window_draws_per_chain=int(rows.shape[0]),
                movements_measured_before_noise_and_residue=True,
                region_threshold_logtau=5.,region_threshold_descriptive_only=True,
                kernels={})
    for kind,name in ((0,'ordinary_hmc'),(7,'exterior_tempered_hmc'),(8,'conditional_heat_refresh')):
        selected=rows[rows[:,:,0]==kind]
        if not len(selected):
            result['kernels'][name]=dict(attempted=0)
            continue
        result['kernels'][name]=dict(attempted=int(len(selected)),
            accepted=int(selected[:,1].sum()),acceptance=float(selected[:,1].mean()),
            mean_acceptance_probability=float(selected[:,2].mean()),
            mean_thermal_diagonal_metric_sqjump=float(selected[:,18].mean()),
            mean_alpha_fraction_sqjump=float(selected[:,19].mean()),
            mean_abs_exterior_logtau_change=float(np.abs(selected[:,26]).mean()),
            exterior_region_crossings=int(selected[:,27].sum()),
            numerically_invalid=int(selected[:,28].sum()),
            abs_endpoint_energy_over_1000=int(selected[:,29].sum()),
            mean_main_seconds=float(selected[:,23].mean()))
    if rows.shape[-1] >= 38:
        attempts=rows[:,:,30] > 0
        selected=rows[attempts]
        result['exterior_transport']=dict(
            attempted=int(attempts.sum()),
            valid=int(selected[:,31].sum()) if len(selected) else 0,
            accepted=int(selected[:,32].sum()) if len(selected) else 0,
            acceptance=float(selected[:,32].mean()) if len(selected) else None,
            mean_acceptance_probability=float(selected[:,33].mean()) if len(selected) else None,
            cross_region_proposals=int(selected[:,34].sum()) if len(selected) else 0,
            region_crossings=int(selected[:,35].sum()) if len(selected) else 0,
            seconds=float(selected[:,36].sum()) if len(selected) else 0.)
        result['fast_region_fraction_by_chain']=rows[:,:,37].mean(axis=0).tolist()
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def replace_closed(temporary, destination):
    deadline = time.monotonic() + 30
    while True:
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)


def atomic_json(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    replace_closed(temporary, path)


def atomic_npz(path, **arrays):
    temporary = path.with_name(path.stem + '.tmp.npz')
    with temporary.open('wb') as handle:
        # Frequent checkpoints are lossless uncompressed ZIP members.
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    replace_closed(temporary, path)


def load_closed(path):
    """Detach every array and close the archive before Windows replacements."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def numerical_hashes():
    return {path.name: sha256(path) for path in sorted((ROOT / 'src').glob('*.py'))}


def args_for(model):
    m=model.base;d=m.d
    return (m.mean,m.bi,m.wi,d['boundaries'],d['g_hvac'],d['tsupply'],d['gains'],d['ginf'],float(d['y'][0]),d['y'],m.idx,m.contiguous,1.)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('period', choices=['july'])
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--minimum-retained',type=int,default=5000)
    parser.add_argument('--initial-map-iterations',type=int,default=1500)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--warmup', type=int, default=12000)
    parser.add_argument('--settling-initial', type=int, default=8000)
    parser.add_argument('--settling-max', type=int, default=20000)
    parser.add_argument('--settling-check-every', type=int, default=4000)
    parser.add_argument('--settling-burnin', type=int, default=4000)
    parser.add_argument('--settling-window', type=int, default=4000)
    parser.add_argument('--settling-confirmations', type=int, default=2)
    parser.add_argument('--max-retained', type=int, default=200000,
                        help='Finite absolute per-chain budget; exits if convergence has not passed')
    parser.add_argument('--seed', type=int, default=20261018)
    parser.add_argument('--particles', type=int, default=512)
    parser.add_argument('--hmc-weight', type=float, default=0.65)
    parser.add_argument('--bridge-weight', type=float, default=0.25)
    parser.add_argument('--bridge-temperature', type=float, default=16.)
    parser.add_argument('--bridge-step-ratio', type=float, default=0.75)
    parser.add_argument('--target-acceptance', type=float, default=0.95)
    parser.add_argument('--metric-shrinkage', type=float, default=0.20)
    parser.add_argument('--trajectory-min', type=float, default=0.4)
    parser.add_argument('--trajectory-max', type=float, default=1.2)
    parser.add_argument('--max-leapfrog', type=int, default=1024)
    parser.add_argument('--residue-substeps', type=int, default=8)
    parser.add_argument('--transport-every',type=int,default=5)
    parser.add_argument('--transport-fit-window',type=int,default=4000)
    parser.add_argument('--transport-min-count',type=int,default=64)
    parser.add_argument('--transport-shrinkage',type=float,default=.2)
    a = parser.parse_args(argv)
    windows = ([4] if a.smoke else warmup_windows(a.warmup))
    if not (0 < a.hmc_weight < 1 and 0 < a.bridge_weight < 1 - a.hmc_weight
            and np.isfinite(a.bridge_temperature) and a.bridge_temperature >= 1
            and np.isfinite(a.bridge_step_ratio) and 0 < a.bridge_step_ratio <= 1
            and 0 < a.target_acceptance < 1 and 0 < a.metric_shrinkage <= 1
            and 0 < a.trajectory_min <= a.trajectory_max
            and a.transport_every >= 1 and a.transport_fit_window >= 100
            and a.transport_min_count >= 8 and 0 < a.transport_shrinkage <= 1
            and a.max_leapfrog >= (2 if a.smoke else 16) and a.particles >= 2 and a.max_retained > 0
            and 0<a.settling_initial<=a.settling_max and a.settling_check_every>0
            and (a.settling_max-a.settling_initial)%a.settling_check_every==0):
        raise ValueError('Invalid tuned sampler settings')
    from profile_heat import ProfileHeatModel as HeatModel
    from profile_coordinates import to_raw, from_raw, prior
    from profile_integrator import profile_trajectory as trajectory
    from exterior_bridge import tempered_profile_trajectory as bridge_trajectory
    from residue_transition import transition as residue_transition
    from exterior_transport import fit as fit_transport, transition as transport_transition, from_arrays as transport_from_arrays
    from global_transport import MixtureTransport
    from settling_policy import SettlingPolicy
    from production_health import production_health
    policy = None if a.smoke else SettlingPolicy(
        a.settling_burnin, a.settling_window, a.settling_confirmations,a.settling_max)
    if not a.smoke and (a.settling_initial != a.settling_burnin+a.settling_window
                        or a.settling_check_every != a.settling_window):
        raise ValueError('Assessment schedule must match the prespecified burn-in/window policy')
    out = (a.output_dir or ROOT / 'results/wide_case5').resolve()
    if (out/'configuration.json').exists() and not a.resume:
        raise RuntimeError('A saved run already exists. Use RESUME_CASE5.bat explicitly; START never resumes or overwrites.')
    if a.resume and not (out/'configuration.json').exists():
        raise RuntimeError('No Case 5 run exists to resume. Use START_CASE5.bat.')
    out.mkdir(parents=True, exist_ok=True)
    stop_path = ROOT / 'CASE5_STOP_REQUESTED'
    local_stop = [False]
    def request_stop(signum, frame):
        local_stop[0] = True
    if hasattr(signal, 'SIGINT'):
        signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, request_stop)
    def stopping():
        return local_stop[0] or stop_path.exists()
    model = HeatModel(a.period)
    fine = model.model
    args = args_for(fine)
    configuration_settings = dict(
        period=a.period, warmup=a.warmup, seed=a.seed, particles=a.particles,
        hmc_weight=a.hmc_weight, bridge_weight=a.bridge_weight,
        bridge_temperature=a.bridge_temperature,
        bridge_step_ratio=a.bridge_step_ratio,
        target_acceptance=a.target_acceptance, metric_shrinkage=a.metric_shrinkage,
        trajectory_min=a.trajectory_min, trajectory_max=a.trajectory_max,
        max_leapfrog=a.max_leapfrog, residue_substeps=a.residue_substeps,
        minimum_retained=a.minimum_retained,
        initial_map_iterations=a.initial_map_iterations,smoke=a.smoke,
        transport_every=a.transport_every,transport_fit_window=a.transport_fit_window,
        transport_min_count=a.transport_min_count,transport_shrinkage=a.transport_shrinkage)
    configuration_settings.update(settling_initial=a.settling_initial,
                                  settling_max=a.settling_max,
                                  settling_check_every=a.settling_check_every,
                                  settling_burnin=a.settling_burnin,
                                  settling_window=a.settling_window,
                                  settling_confirmations=a.settling_confirmations)
    hashes = numerical_hashes()
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (
        ROOT / 'inputs/data.npz', ROOT / 'inputs/reference.json')}
    if (out / 'configuration.json').exists():
        config = json.loads((out / 'configuration.json').read_text(encoding='utf-8'))
        if (config['settings'] != configuration_settings
                or config['numerical_source_hashes'] != hashes
                or config['input_sha256'] != input_hashes):
            raise RuntimeError('Frozen settings, numerical source or input changed; use a new output folder')
        if config['initial_states_sha256'] != sha256(out / 'initial_states.npz'):
            raise RuntimeError('Frozen initial states changed')
        initial = load_closed(out / 'initial_states.npz')
        q, covariance = initial['position'], initial['initial_position_covariance']
    else:
        from july_initialization import fresh_positions
        print(json.dumps(dict(period='july',stage='fresh_initialization',chains=4,old_draws_imported=0)),flush=True)
        q, covariance, provenance = fresh_positions(fine,a.seed,a.initial_map_iterations,a.smoke)
        metric, _ = momentum_metric(covariance, a.metric_shrinkage)
        atomic_npz(out / 'initial_states.npz', position=q, metric=metric,
                   initial_position_covariance=covariance)
        config = dict(settings=configuration_settings, package_id='HIBO_LabB_SingleSP_Initial_v3',
                      diagnostic_definition=diagnostic_definition(), runtime_environment=runtime_environment(), method='Case 5 HMC/bridge/heat/residue plus frozen targeted exterior-capacity transport with exact multiple-try heat refresh',
                      numerical_source_hashes=hashes, input_sha256=input_hashes,
                      initial_states_sha256=sha256(out / 'initial_states.npz'),
                      independent_chains=CHAINS, n_R_free=17, n_C_free=8,
                      n_heat_free=28, n_error_free=2, n_parameters=55, n_reported=59,
                      thermal_model_and_priors_unchanged=True, likelihood_interval_minutes=5,
                      likelihood_formula_unchanged_from_case5=True,
                      data_source='New July single-setpoint CSV',
                      physical_input_seconds=60,integration_substep_seconds=30,
                      conditioning_first_hours=24,conditioning_proves_equilibration=False,
                      AR1_prior_timebase_minutes=60,AR1_interpolation='stationary positive-correlation Gaussian semigroup',
                      production_adaptation=False,
                      retained_restarts_from_zero=True, source_rng_continued=False,
                      profile_coordinates=True, weighted_profile_coordinates=False,
                      warmup_metric_windows=windows, warmup_final_buffer=a.warmup-windows[-1],
                      thresholds=dict(rhat=1.10, bulk_ess=400, tail_ess=400),
                      production_numerical_health_required=True,
                      numerical_health_policy='Inherited settling engineering screen applied also to all retained and latest4000 production draws; per-chain ordinary-HMC invalid-or-large-energy fraction <=1%',
                      sampler_setting_rationale='target acceptance .95: standard HMC stability adjustment; warmup12000: longer discarded metric/step adaptation, not a posterior or prior change',
                      tail_ess_quantiles=[0.05,0.95],
                      discarded_frozen_settling=True,
                      settling_gate_window='Prespecified discarded frozen burn-in followed by nonoverlapping assessment windows; consecutive windows and their pool must pass',
                      settling_policy=policy.metadata() if policy else dict(smoke_only=True),
                      settling_gate_is_equilibration_guarantee=False,
                      momentum_metric_convention='M=inverse(position covariance); p~N(0,M)',
                      heat_prior='Original four Dirichlet(1,...,1), unchanged',
                      thermal_proposal='Rank-one exterior kinetic-direction heating/cooling trajectory; original cold endpoint Hamiltonian acceptance',
                      bridge_direction='Momentum coordinate 18; velocity direction inverse_metric[:,18] couples all coordinates',
                      bridge_step_adaptation='Fixed ratio of ordinary-HMC integration step; tempered work rejection never drives step adaptation',
                      exterior_region_logtau_threshold_descriptive_only=5.,
                      legacy_near_inactive_thermal_blocks_removed=True,
                      intermediate_heated_states_retained=False,
                      transport_coordinates=['fastest_exterior_log_tau','fastest_exterior_log_effective_C_over_air_C','log_air_C'],
                      transport_target_unchanged=True, transport_fit_source='this run discarded warmup only',
                      transport_applies_to_initial_only=True,
                      transport_region_threshold_is_not_a_parameter_bound=True,
                      transport_production_adaptation=False,
                      statistic_names=STATISTIC_NAMES, stop_at_first_N=True, run_2N=False,
                      **provenance)
        atomic_json(out / 'configuration.json', config)
    metric, inverse = momentum_metric(covariance, a.metric_shrinkage)
    chol = np.linalg.cholesky(metric)
    # Initialize independent random streams for a fresh run.
    rngs = [np.random.default_rng([a.seed, a.period == 'july', c]) for c in range(CHAINS)]
    labels = [np.random.default_rng([a.seed + 1, a.period == 'july', c]) for c in range(CHAINS)]
    steps = np.full(CHAINS, 0.05)
    averaging = [DualAveraging(step, a.target_acceptance) for step in steps]
    bridge_steps = steps*a.bridge_step_ratio
    lp = np.asarray([fine.logpost(value) for value in q])
    if not np.isfinite(lp).all():
        raise FloatingPointError('An initial endpoint has nonfinite exact target density')
    warmup_completed, completed, N = 0, 0, None
    production_initialized = False
    frozen_ready, settling_completed, settling_passed = False, 0, False
    settling_records, settling_statistics, settling_history = [], [], []
    warmup_records, warmup_statistics = [], []
    metric_window_start = 0
    history, metric_history = [], []
    transport=None
    transport_history=[]
    def transport_arrays():
        return dict(transport_means=np.empty((0,3)) if transport is None else transport.means,
                    transport_covariances=np.empty((0,3,3)) if transport is None else transport.covariances,
                    transport_weights=np.empty(0) if transport is None else transport.weights)
    def restore_transport(saved):
        if 'transport_means' not in saved:
            raise RuntimeError('v2 transport checkpoint fields missing; foreign checkpoints cannot be resumed')
        return transport_from_arrays(saved)
    def update_transport():
        nonlocal transport
        candidate, information=fit_transport(np.asarray(warmup_records),
            window=a.transport_fit_window,min_count=a.transport_min_count,
            shrinkage=a.transport_shrinkage)
        if candidate is not None:transport=candidate
        information.update(completed=warmup_completed,
                           kept_previous=candidate is None and transport is not None)
        transport_history.append(information)
        atomic_json(out/'transport_fit_history.json',transport_history)
        print(json.dumps(dict(stage='discarded_warmup_transport_fit',**information)),flush=True)
    us = np.empty((CHAINS, 0, DIMENSION))
    ws = np.empty_like(us)
    logs = np.empty((CHAINS, 0))
    stats = np.empty((CHAINS, 0, len(STATISTIC_NAMES)))
    start = time.monotonic()
    def status(stage, **information):
        payload = dict(period=a.period, stage=stage, N=N,
                       elapsed_seconds=time.monotonic()-start, **information)
        payload['diagnostic_definition'] = diagnostic_definition()
        atomic_json(out / 'status.json', payload)
        console_payload = {key: value for key, value in payload.items() if key != 'diagnostic_definition'}
        for key in ('physical', 'ordered'):
            if isinstance(console_payload.get(key), dict):
                console_payload[key] = {name: value for name, value in console_payload[key].items()
                                        if name not in ('diagnostic_definition', 'per_parameter')}
        print(json.dumps(console_payload, allow_nan=False), flush=True)
    def rng_state():
        return dict(rngs=[r.bit_generator.state for r in rngs],
                    label_rngs=[r.bit_generator.state for r in labels])
    def restore_rng(state):
        for c in range(CHAINS):
            rngs[c].bit_generator.state = state['rngs'][c]
            labels[c].bit_generator.state = state['label_rngs'][c]
    def checked_density():
        exact = np.asarray([fine.logpost(value) for value in q])
        discrepancy = float(np.max(np.abs(exact - lp)))
        if not np.isfinite(exact).all() or discrepancy >= 1e-5:
            raise AssertionError(('Cached exact target density discrepancy', discrepancy))
        lp[:] = exact
        return discrepancy
    def warmup_checkpoint():
        discrepancy = checked_density()
        state = dict(completed=warmup_completed, diagnostic_definition=diagnostic_definition(), metric_window_start=metric_window_start,
                     averaging=[value.to_dict() for value in averaging],
                     metric_history=metric_history, transport_history=transport_history, position_logdensity_check_error=discrepancy,
                     **rng_state())
        atomic_npz(out / 'warmup_checkpoint.npz', position=q, position_logp=lp,
                   steps=steps, bridge_steps=bridge_steps, metric=metric, inverse_metric=inverse,
                   discarded_w=np.asarray(warmup_records),
                   discarded_statistics=np.asarray(warmup_statistics),**transport_arrays(),
                   checkpoint_state_json=np.asarray(json.dumps(state)))
        atomic_json(out / 'warmup_checkpoint_state.json', state)
    def production_checkpoint():
        discrepancy = checked_density()
        state = dict(retained_per_chain=completed, N=N, history=history, diagnostic_definition=diagnostic_definition(),
                     position_logdensity_check_error=discrepancy, **rng_state())
        atomic_npz(out / 'checkpoint.npz', u=us, w=ws, logp=logs,
                   statistics=stats, position=q, position_logp=lp,
                   steps=steps, bridge_steps=bridge_steps, metric=metric, inverse_metric=inverse,
                   checkpoint_state_json=np.asarray(json.dumps(state)))
        atomic_json(out / 'checkpoint_state.json', state)
        atomic_json(out / 'history.json', history)
    def settling_checkpoint():
        discrepancy = checked_density()
        state = dict(completed=settling_completed, passed=settling_passed, diagnostic_definition=diagnostic_definition(),
                     history=settling_history,position_logdensity_check_error=discrepancy,
                     frozen_steps=steps.tolist(),frozen_bridge_steps=bridge_steps.tolist(),**rng_state())
        atomic_npz(out/'settling_checkpoint.npz',position=q,position_logp=lp,
                   steps=steps,bridge_steps=bridge_steps,metric=metric,inverse_metric=inverse,
                   discarded_w=np.asarray(settling_records),
                   discarded_statistics=np.asarray(settling_statistics),
                   checkpoint_state_json=np.asarray(json.dumps(state)))
        atomic_json(out/'settling_state.json',state)
    if (out / 'checkpoint.npz').exists():
        saved = load_closed(out / 'checkpoint.npz')
        state = json.loads(str(saved['checkpoint_state_json']))
        us, ws, logs, stats = (saved[name] for name in ('u', 'w', 'logp', 'statistics'))
        q, lp, steps = (saved[name] for name in ('position', 'position_logp', 'steps'))
        metric, inverse = saved['metric'], saved['inverse_metric']
        chol, bridge_steps = np.linalg.cholesky(metric), saved['bridge_steps']
        completed, N, history = us.shape[1], state['N'], state['history']
        if completed != state['retained_per_chain']:
            raise ValueError('Checkpoint record count mismatch')
        if sha256(out / 'frozen_kernel.npz') != config['frozen_kernel_sha256']:
            raise RuntimeError('Frozen kernel changed')
        restore_rng(state)
        warmup_completed = a.warmup
        production_initialized = True
        frozen_ready = settling_passed = True
        status('resume', retained_per_chain=completed)
    elif (out/'settling_checkpoint.npz').exists():
        saved=load_closed(out/'settling_checkpoint.npz')
        state=json.loads(str(saved['checkpoint_state_json']))
        q,lp,steps=(saved[name] for name in ('position','position_logp','steps'))
        metric,inverse=saved['metric'],saved['inverse_metric']
        chol,bridge_steps=np.linalg.cholesky(metric),saved['bridge_steps']
        settling_completed=int(state['completed'])
        settling_passed=bool(state['passed'])
        settling_history=state['history']
        settling_records=list(saved['discarded_w'])
        settling_statistics=list(saved['discarded_statistics'])
        if settling_completed!=len(settling_records) or settling_completed!=len(settling_statistics):
            raise ValueError('Frozen-settling checkpoint record count mismatch')
        if sha256(out/'frozen_kernel.npz')!=config['frozen_kernel_sha256']:
            raise RuntimeError('Frozen settling kernel changed')
        restore_rng(state)
        warmup_completed=a.warmup
        frozen_ready=True
        status('frozen_settling_resume',completed=settling_completed,
               maximum=a.settling_max,retained_per_chain=0)
    elif (out / 'warmup_checkpoint.npz').exists():
        saved = load_closed(out / 'warmup_checkpoint.npz')
        state = json.loads(str(saved['checkpoint_state_json']))
        q, lp, steps = (saved[name] for name in ('position', 'position_logp', 'steps'))
        metric, inverse = saved['metric'], saved['inverse_metric']
        chol = np.linalg.cholesky(metric)
        bridge_steps=saved['bridge_steps']
        warmup_completed = state['completed']
        metric_window_start = state['metric_window_start']
        metric_history = state['metric_history']
        transport_history=state['transport_history']
        transport=restore_transport(saved)
        averaging = [DualAveraging.from_dict(value) for value in state['averaging']]
        warmup_records = list(saved['discarded_w'])
        warmup_statistics = list(saved['discarded_statistics'])
        restore_rng(state)
        status('warmup_resume', completed=warmup_completed, total=a.warmup)
    if frozen_ready:
        transport=restore_transport(load_closed(out/'frozen_kernel.npz'))
    if N is not None:
        status('complete', retained_per_chain=completed)
        return
    trajectory(q[0], np.zeros(DIMENSION), 0.001, 1, inverse, args)
    bridge_trajectory(q[0], np.zeros(DIMENSION), 0.001, 2, inverse, args, 1.)
    def one(c, iteration, warm=False):
        t0 = time.monotonic()
        rng = rngs[c]
        result = np.zeros(len(STATISTIC_NAMES))
        kind, accepted, probability = 0, 0.0, 0.0
        old = q[c].copy()
        old_alpha=fine.base.alpha(to_raw(old))
        if warm:
            bridge_steps[c]=steps[c]*a.bridge_step_ratio
        branch=rng.random()
        if branch < a.hmc_weight+a.bridge_weight:
            is_bridge=branch >= a.hmc_weight
            kind=7 if is_bridge else 0
            momentum = chol @ rng.normal(size=DIMENSION)
            requested_time = rng.uniform(a.trajectory_min, a.trajectory_max)
            step=bridge_steps[c] if is_bridge else steps[c]
            requested_length = max(1, int(round(requested_time/step)))
            if is_bridge:
                # The fixed, state-independent temperature draw is part of the
                # proposal. Intermediate heated states are never retained.
                peak=float(np.exp(rng.uniform(np.log(max(1.,a.bridge_temperature/4.)),
                                               np.log(a.bridge_temperature))))
                requested_length=max(2,2*int(round(requested_length/2)))
                length=min(a.max_leapfrog-a.max_leapfrog%2,max(4,requested_length))
                proposal,pnew,value,valid=bridge_trajectory(q[c],momentum,step,length,inverse,args,peak)
                result[25]=peak
            else:
                length = min(a.max_leapfrog, max(4, requested_length))
                proposal, pnew, value, valid = trajectory(q[c], momentum, step, length, inverse, args)
            delta = float(value-lp[c] - .5*(pnew@inverse@pnew-momentum@inverse@momentum)) if valid else -np.inf
            probability = float(np.exp(min(0., delta))) if np.isfinite(delta) else 0.0
            accepted = float(np.log(rng.random()) < delta)
            if accepted:
                q[c], lp[c] = proposal, value
            result[5] = float(not valid or not np.isfinite(delta) or abs(delta)>1000)
            result[28]=float(not valid or not np.isfinite(delta))
            result[29]=float(np.isfinite(delta) and abs(delta)>1000)
            result[11:16] = [-delta if np.isfinite(delta) else 1e300,
                             length, step, length*step,
                             float(requested_length>a.max_leapfrog)]
            if warm:
                if not is_bridge:
                    steps[c] = averaging[c].update(probability)
                    bridge_steps[c]=steps[c]*a.bridge_step_ratio
        else:
            kind=8
            theta = to_gap(q[c])[:25]
            # Refresh heat allocations conditional on the current thermal state.
            q[c], lp[c], information = heat_transition(model, q[c], theta, 0., rng, a.particles)
            accepted, probability = information['accepted'], information['probability']
            result[3:5] = [information['importance_ess'], information['maximum_weight']]
            result[6] = 1.0
        main_displacement=q[c]-old
        result[18]=float(np.mean(main_displacement[:25]**2*np.diag(metric)[:25]))
        result[19]=float(np.mean((fine.base.alpha(to_raw(q[c]))-old_alpha)**2))
        result[23]=time.monotonic()-t0
        result[26]=q[c,18]-old[18]
        result[27]=float((q[c,18]<5.)!=(old[18]<5.))
        auxiliary_start=time.monotonic()
        old_noise=q[c,25:27].copy()
        if iteration % 5 == 0:
            raw = to_raw(q[c])
            noise_statistics = fine.base.stats(fine.base.native(raw))
            noise = q[c, 25:27].copy()
            fn = lambda values: fine.base.noise_logpost(values, noise_statistics)
            before = fn(noise)
            value = before
            for coordinate in rng.permutation(2):
                noise, value, _, _ = coordinate_slice(noise, value, fn, rng, [coordinate], width=.15, max_steps=20)
            q[c, 25:27] = noise
            lp[c] += value-before
        result[20]=float(np.mean((q[c,25:27]-old_noise)**2))
        if iteration % 10 == 0:
            priorfn = lambda values: prior(values, fine.base.mean)
            prior_before = priorfn(q[c])
            q[c], prior_after, _, _ = coordinate_slice(q[c], prior_before, priorfn, rng, [15], width=1.5, max_steps=20)
            lp[c] += prior_after-prior_before
        before_residue=q[c].copy()
        before_residue_alpha=fine.base.alpha(to_raw(before_residue))
        q[c], lp[c], residue = residue_transition(fine, q[c], lp[c], rng, a.residue_substeps)
        residue_displacement=q[c]-before_residue
        result[21]=float(np.mean(residue_displacement[:25]**2*np.diag(metric)[:25]))
        result[22]=float(np.mean((fine.base.alpha(to_raw(q[c]))-before_residue_alpha)**2))
        if transport is not None and iteration % a.transport_every == 0:
            move_start=time.monotonic()
            q[c],lp[c],movement=transport_transition(model,q[c],lp[c],rng,transport,a.particles)
            result[30:36]=[1.,movement['valid'],movement['accepted'],movement['probability'],
                           movement['cross_region_proposal'],movement['region_crossing']]
            result[36]=time.monotonic()-move_start
        result[37]=float(q[c,18]<5.)
        result[24]=time.monotonic()-auxiliary_start
        # Slice/block updates add density differences. Refresh the exact target
        # every transition so a checkpoint boundary never changes rounding or
        # subsequent acceptance decisions relative to uninterrupted execution.
        lp[c]=fine.logpost(q[c])
        if not np.isfinite(lp[c]):raise FloatingPointError('Nonfinite exact endpoint density')
        displacement = q[c]-old
        result[:3] = [kind, accepted, probability]
        result[7:11] = [residue['attempted'], residue['valid'], residue['accepted'], residue['nontrivial']]
        result[16] = float(displacement@metric@displacement/DIMENSION)
        result[17] = time.monotonic()-t0
        return result
    with ThreadPoolExecutor(CHAINS) as pool:
        while warmup_completed < a.warmup:
            if stopping():
                warmup_checkpoint()
                status('stopped', stage_detail='discarded_warmup', completed=warmup_completed)
                return
            row = np.asarray(list(pool.map(lambda c: one(c, warmup_completed, True), range(CHAINS))))
            warmup_records.append(q.copy())
            warmup_statistics.append(row)
            warmup_completed += 1
            if warmup_completed in windows:
                update_transport()
                positions = np.asarray(warmup_records[metric_window_start:])
                covariance_estimate = pooled_covariance(positions, inverse)
                metric, inverse = momentum_metric(covariance_estimate, a.metric_shrinkage)
                chol = np.linalg.cholesky(metric)
                metric_window_start = warmup_completed
                # Retune steps under the changed metric; adaptation cannot continue
                # into retained production. The last third has a fixed metric.
                previous = np.asarray([value.final_step() for value in averaging])
                steps = np.clip(previous*.5, 1e-5, .5)
                averaging = [DualAveraging(step, a.target_acceptance) for step in steps]
                bridge_steps=steps*a.bridge_step_ratio
                metric_history.append(dict(completed=warmup_completed,
                    metric_condition_number=float(np.linalg.cond(metric)),
                    covariance_positions=int(len(positions)*CHAINS),
                    reset_steps=steps.tolist(),reset_bridge_steps=bridge_steps.tolist()))
            if warmup_completed % 100 == 0 or warmup_completed == a.warmup:
                warmup_checkpoint()
                rows = np.asarray(warmup_statistics[-100:])
                hmc = rows[:, :, 0]==0
                atomic_json(out/'kernel_summary.json',dict(stage='discarded_adaptive_warmup',
                            completed=warmup_completed,**kernel_summary(rows)))
                status('warmup_progress', completed=warmup_completed, total=a.warmup,
                       steps=steps.tolist(), bridge_steps=bridge_steps.tolist(),
                       hmc_acceptance=float(rows[:, :, 1][hmc].mean()) if np.any(hmc) else None,
                       bridge_acceptance=float(rows[:,:,1][rows[:,:,0]==7].mean()) if np.any(rows[:,:,0]==7) else None,
                       large_energy_fraction=float(rows[:, :, 5][hmc].mean()) if np.any(hmc) else None,
                       length_capped_fraction=float(rows[:, :, 15][hmc].mean()) if np.any(hmc) else None,
                       metric_condition_number=float(np.linalg.cond(metric)))
        if not production_initialized and not frozen_ready:
            update_transport()
            steps = np.asarray([value.final_step() for value in averaging])
            bridge_steps=steps*a.bridge_step_ratio
            atomic_npz(out/'frozen_kernel.npz',steps=steps,bridge_steps=bridge_steps,
                       metric=metric,inverse_metric=inverse,
                       bridge_temperature=np.array(a.bridge_temperature),**transport_arrays())
            config.update(frozen_steps=steps.tolist(), frozen_bridge_steps=bridge_steps.tolist(),
                          frozen_kernel_sha256=sha256(out/'frozen_kernel.npz'),
                          metric_history=metric_history,transport_fit_history=transport_history,
                          frozen_transport_enabled=transport is not None)
            atomic_json(out / 'configuration.json', config)
            atomic_npz(out / 'warmup_end.npz', position=q, position_logp=lp, steps=steps,
                       bridge_steps=bridge_steps,metric=metric, inverse_metric=inverse)
            frozen_ready=True
            settling_checkpoint()
            status('adaptation_complete',retained_per_chain=0,
                   next_stage='discarded_frozen_settling',frozen_steps=steps.tolist(),
                   frozen_bridge_steps=bridge_steps.tolist(),settling_initial=a.settling_initial,
                   settling_maximum=a.settling_max)
        if not production_initialized:
            from warmup_health import evaluate_settling
            initial_settling=4 if a.smoke else a.settling_initial
            maximum_settling=4 if a.smoke else a.settling_max
            while not settling_passed and settling_completed<maximum_settling:
                if stopping():
                    settling_checkpoint()
                    status('stopped',stage_detail='discarded_frozen_settling',
                           completed=settling_completed,retained_per_chain=0)
                    return
                rows=np.asarray(list(pool.map(
                    lambda c:one(c,a.warmup+settling_completed,False),range(CHAINS))))
                settling_records.append(q.copy())
                settling_statistics.append(rows)
                settling_completed+=1
                if settling_completed%100==0:
                    settling_checkpoint()
                    recent=np.asarray(settling_statistics[-100:])
                    hmc=recent[:,:,0]==0
                    mobility=kernel_summary(recent)
                    atomic_json(out/'kernel_summary.json',dict(stage='discarded_frozen_settling',
                                completed=settling_completed,**mobility))
                    status('frozen_settling',completed=settling_completed,
                           initial=initial_settling,maximum=maximum_settling,
                           retained_per_chain=0,
                           hmc_acceptance=float(recent[:,:,1][hmc].mean()) if hmc.any() else None,
                           bridge_acceptance=mobility['kernels']['exterior_tempered_hmc'].get('acceptance'),
                           transport=mobility.get('exterior_transport'),
                           fast_region_fraction_by_chain=mobility.get('fast_region_fraction_by_chain'),
                           length_capped_fraction=float(recent[:,:,15][hmc].mean()) if hmc.any() else None)
                check=False if a.smoke else policy.is_assessment(settling_completed)
                if a.smoke and settling_completed==maximum_settling:
                    check=True
                if check:
                    if a.smoke:
                        audit=dict(passed=True,smoke_only=True,
                                   inference_is_not_authorized_by_this_gate=True)
                    else:
                        lower, upper=policy.bounds(settling_completed)
                        audit=evaluate_settling(fine,
                            np.swapaxes(np.asarray(settling_records[lower:upper]),0,1),
                            np.swapaxes(np.asarray(settling_statistics[lower:upper]),0,1),
                            a.seed+100000+settling_completed)
                        audit.update(window_passed=bool(audit['passed']),
                                     assessment_start_iteration=lower+1,
                                     assessment_end_iteration=upper,
                                     policy=policy.metadata())
                        audit['kernel_movement']=kernel_summary(np.asarray(settling_statistics[lower:upper]))
                        candidate=dict(passed=audit['window_passed'],completed=settling_completed)
                        ready=policy.candidate_ready(settling_history,candidate)
                        if ready:
                            pool_lower=settling_completed-policy.window*policy.confirmations
                            pooled=evaluate_settling(fine,
                                np.swapaxes(np.asarray(settling_records[pool_lower:upper]),0,1),
                                np.swapaxes(np.asarray(settling_statistics[pool_lower:upper]),0,1),
                                a.seed+200000+settling_completed)
                            audit['pooled_confirmation']=pooled
                            audit['passed']=bool(pooled['passed'])
                            if not audit['passed']:
                                audit['blocking_failures'].extend(pooled['blocking_failures'])
                        else:
                            audit['passed']=False
                            if audit['window_passed']:
                                audit['blocking_failures'].append(
                                    'Awaiting the prespecified consecutive nonoverlapping assessment confirmations')
                    settling_passed=bool(audit['passed'])
                    settling_history.append(dict(completed=settling_completed,**audit))
                    atomic_json(out/'settling_diagnostics.json',settling_history[-1])
                    settling_checkpoint()
                    status('frozen_settling_gate',completed=settling_completed,
                           maximum=maximum_settling,retained_per_chain=0,
                           passed=settling_passed,
                           blocking_failures=audit.get('blocking_failures',[]),
                           remaining_budget=maximum_settling-settling_completed)
            if not settling_passed:
                settling_checkpoint()
                status('settling_gate_failed',completed=settling_completed,
                       maximum=maximum_settling,retained_per_chain=0,
                       explanation='Discarded fixed-kernel settling budget exhausted; no retained production was started.')
                return
            config.update(discarded_settling_completed=settling_completed,
                          discarded_settling_history=settling_history,
                          all_production_starts_after_settling_gate=True)
            atomic_json(out/'configuration.json',config)
            atomic_npz(out/'settling_end.npz',position=q,position_logp=lp,
                       steps=steps,bridge_steps=bridge_steps,metric=metric,inverse_metric=inverse)
            production_initialized=True
            production_checkpoint()
            status('warmup_complete',retained_per_chain=0,
                   discarded_adaptive=a.warmup,discarded_frozen_settling=settling_completed,
                   frozen_steps=steps.tolist(),frozen_bridge_steps=bridge_steps.tolist())
        while N is None:
            if stopping():
                production_checkpoint()
                status('stopped', retained_per_chain=completed)
                return
            if a.max_retained and completed >= a.max_retained:
                production_checkpoint()
                status('smoke_complete_no_inference' if a.smoke else 'search_budget_reached', retained_per_chain=completed)
                return
            count = min(250, 500-completed%500)
            if a.max_retained:
                count = min(count, a.max_retained-completed)
            block_start = time.monotonic()
            ublock, wblock, lblock, sblock = [], [], [], []
            for offset in range(count):
                rows = np.asarray(list(pool.map(lambda c: one(c, completed+offset), range(CHAINS))))
                restored, labeled_lp = np.empty((CHAINS, DIMENSION)), np.empty(CHAINS)
                for c in range(CHAINS):
                    restored[c], labeled_lp[c], _ = fine.draw_labels(q[c], labels[c], lp[c])
                ublock.append(restored)
                wblock.append(q.copy())
                lblock.append(labeled_lp)
                sblock.append(rows)
                if stopping():
                    break
            us = np.concatenate([us, np.swapaxes(np.asarray(ublock), 0, 1)], axis=1)
            ws = np.concatenate([ws, np.swapaxes(np.asarray(wblock), 0, 1)], axis=1)
            logs = np.concatenate([logs, np.asarray(lblock).T], axis=1)
            stats = np.concatenate([stats, np.swapaxes(np.asarray(sblock), 0, 1)], axis=1)
            completed += len(ublock)
            recent = np.asarray(sblock)
            hmc = recent[:, :, 0]==0
            thermal = ~hmc
            summary = dict(retained_per_chain=completed, diagnostic_definition=diagnostic_definition(),
                transition_block_seconds=time.monotonic()-block_start,
                hmc_acceptance=float(recent[:, :, 1][hmc].mean()) if np.any(hmc) else None,
                hmc_large_energy_fraction=float(recent[:, :, 5][hmc].mean()) if np.any(hmc) else None,
                hmc_leapfrog_mean=float(recent[:, :, 12][hmc].mean()) if np.any(hmc) else None,
                hmc_length_capped_fraction=float(recent[:, :, 15][hmc].mean()) if np.any(hmc) else None,
                bridge_acceptance=float(recent[:,:,1][recent[:,:,0]==7].mean()) if np.any(recent[:,:,0]==7) else None,
                conditional_heat_acceptance=float(recent[:,:,1][recent[:,:,0]==8].mean()) if np.any(recent[:,:,0]==8) else None,
                conditional_importance_ess_median=float(np.median(recent[:,:,3][recent[:,:,0]==8])) if np.any(recent[:,:,0]==8) else None,
                metric_condition_number=float(np.linalg.cond(metric)))
            mobility=kernel_summary(recent)
            summary['kernel_movement']=mobility
            atomic_json(out/'kernel_summary.json',dict(stage='retained_production',completed=completed,**mobility))
            if not a.smoke and completed >= a.minimum_retained and completed % 500 == 0:
                diagnostic_start = time.monotonic()
                physical = diagnostics(fine.base, us)
                ordered = canonical_diagnostics(to_v(to_raw(ws)))
                numerical=production_health(stats)
                atomic_json(out/'production_health.json',numerical)
                passed = physical['passed'] and ordered['passed'] and numerical['passed']
                summary.update(physical={key:value for key,value in physical.items() if key!='per_parameter'},
                               ordered=ordered, numerical_health=numerical, passed=passed,
                               diagnostic_seconds=time.monotonic()-diagnostic_start)
                atomic_json(out / 'diagnostics.json', physical)
                atomic_json(out / 'ordered_diagnostics.json', ordered)
                from july_export import checkpoint_metrics
                summary['temperature_metrics']=checkpoint_metrics(fine.base,us,logs)
                atomic_json(out/'temperature_metrics.json',summary['temperature_metrics'])
                if passed:
                    N = completed
                    atomic_npz(out / 'N.npz', u=us, w=ws, logp=logs)
            history.append(summary)
            checkpoint_start = time.monotonic()
            production_checkpoint()
            summary['checkpoint_seconds'] = time.monotonic()-checkpoint_start
            atomic_json(out / 'history.json', history)
            status('complete' if N is not None else 'sampling', **summary)
        # Stop at the first complete diagnostic pass.
        status('complete', retained_per_chain=completed, first_passing_N=N)


if __name__ == '__main__':
    main()
