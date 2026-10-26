"""Read-only replay of completed v2 trials under the corrected v3 adoption rule.

This is not MCMC checkpoint migration. No chains are loaded, re-diagnosed, pooled,
or restarted. Existing diagnostics and fits are retained; selection is recomputed
from complete frozen-parent waves and predictions are checked deterministically.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

import numpy as np

from .topology import (candidate_pairs, counts, merge_pair, model_hash,
                       validate_model, apply_parameters)
from .ranking import parameter_changes, rank_trials
from .numerics import load_data, simulate
from .analysis import prediction_metrics

SOURCE_FILES = (
    'run_identity.json', 'workflow_state.json', 'all_trials.json',
    'models/M_00_17R8C.json', 'summary.json',
)
_V2_RANKING_SHA256 = 'b3f43cd9bcfc5cfcd6e2a7bd16f8f1ab4fdf22581f4d03024d4d3cc94d4c3a2e'
_UNCHANGED_CODE = ('topology.py', 'numerics.py', 'sampling.py', 'analysis.py')


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _source_payload(source):
    """Read only five bounded JSON members; never extract an arbitrary ZIP."""
    source = Path(source).expanduser().resolve()
    payload, digests = {}, {}
    if source.is_dir():
        if not (source / 'workflow_state.json').is_file() and (source / 'results/workflow_state.json').is_file():
            source = source / 'results'
        if (source / 'RUNNING.lock').exists():
            raise ValueError('Source may still be running. Export a completed snapshot first.')
        for name in SOURCE_FILES:
            p = source / name
            if not p.is_file():
                raise ValueError(f'Completed source is missing {name}')
            if p.stat().st_size > 100 * 1024**2:
                raise ValueError(f'Unexpectedly large JSON source: {name}')
            raw = p.read_bytes()
            payload[name], digests[name] = json.loads(raw), _sha_bytes(raw)
        source_kind = 'directory'
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            names = z.namelist()
            prefixes = [n[:-len('workflow_state.json')] for n in names
                        if n.endswith('workflow_state.json')]
            if len(prefixes) != 1:
                raise ValueError('Source ZIP must contain exactly one workflow_state.json')
            prefix = prefixes[0]
            if prefix + 'RUNNING.lock' in names:
                raise ValueError('Source ZIP contains a running lock; use a completed export.')
            for name in SOURCE_FILES:
                full = prefix + name
                if names.count(full) != 1:
                    raise ValueError(f'Source ZIP is missing or duplicates {name}')
                info = z.getinfo(full)
                if info.file_size > 100 * 1024**2:
                    raise ValueError(f'Unexpectedly large JSON source: {name}')
                raw = z.read(full)
                payload[name], digests[name] = json.loads(raw), _sha_bytes(raw)
        source_kind = 'zip'
    else:
        raise ValueError('Source must be a completed results directory or results ZIP')
    return source, source_kind, payload, digests


def _same(a, b, context):
    """Compare regenerated scientific descriptors with a tight roundoff tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            raise ValueError(f'{context}: dictionary fields differ')
        for key in a:
            _same(a[key], b[key], f'{context}.{key}')
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise ValueError(f'{context}: sequence lengths differ')
        for index, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{context}[{index}]')
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{context}: numerical values differ')
    elif a != b:
        raise ValueError(f'{context}: values differ')


def _validate_source(root, config, payload):
    from .workflow import stable_hash, sha, initial_model
    state, identity = payload['workflow_state.json'], payload['run_identity.json']
    rows = payload['all_trials.json']
    if state.get('schema') != 'hibo-workflow-allpairs-rca-v2' or state.get('mode') != 'production' or state.get('status') != 'completed':
        raise ValueError('Replay accepts only completed production v2 results; use normal resume for compatible unfinished runs.')
    if state.get('run_hash') != stable_hash(identity):
        raise ValueError('Source run_identity hash does not match its workflow state')
    if stable_hash(rows) != stable_hash(state.get('rows')):
        raise ValueError('Source all_trials.json disagrees with workflow_state rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Completed source has no candidate records')
    old = identity.get('config', {})
    if old.get('schema') != 'hibo-reduction-allpairs-rca-v2' or old.get('mode') != 'production':
        raise ValueError('Source configuration is not the supported v2 production format')
    for key in ('seed', 'likelihood_minutes', 'validation_rmse_limit_C', 'sampler'):
        if old.get(key) != config.get(key):
            raise ValueError(f'Replay cannot reuse results with changed {key}')
    for key in ('method', 'objectives', 'tie_break'):
        if old.get('selection', {}).get(key) != config.get('selection', {}).get(key):
            raise ValueError(f'Replay cannot change selection {key}')
    if old.get('selection', {}).get('min_pass_candidates') != 2 or config.get('selection', {}).get('min_pass_candidates') != 1:
        raise ValueError('Replay requires the documented v2 minimum-2 to v3 minimum-1 correction')
    files = {str(k).replace('\\', '/'): v for k, v in identity.get('files', {}).items()}
    if len(files) != len(identity.get('files', {})):
        raise ValueError('Ambiguous source identity file paths')
    for p in sorted((Path(root) / 'inputs').glob('*')):
        if p.is_file():
            key = p.relative_to(root).as_posix()
            if files.get(key) != sha(p):
                raise ValueError(f'Source input differs from this package: {key}')
    for name in _UNCHANGED_CODE:
        key = f'src/reduction/{name}'
        if files.get(key) != sha(Path(root) / key):
            raise ValueError(f'Scientific implementation differs from source: {key}')
    if files.get('src/reduction/ranking.py') != _V2_RANKING_SHA256:
        raise ValueError('Source ranking is not the audited v2 implementation')
    m0 = initial_model(root)
    if model_hash(payload['models/M_00_17R8C.json']) != model_hash(m0):
        raise ValueError('Source initial model differs from the packaged Case 5 joint draw')
    if state.get('depth') != len(state.get('path', [])):
        raise ValueError('Source depth/path length mismatch')
    if len({r.get('trial_id') for r in rows}) != len(rows):
        raise ValueError('Duplicate source trial IDs would inflate support')
    summary = payload['summary.json']
    for key, value in (('status', 'completed'), ('n_trials', len(rows)), ('accepted_steps', state['depth'])):
        if summary.get(key) != value:
            raise ValueError(f'Source summary disagrees with state: {key}')
    if summary.get('current_model') != counts(state['current_model']):
        raise ValueError('Source summary model disagrees with state')
    if state['path']:
        previous = next((r for r in rows if r['trial_id'] == state['path'][-1]['trial_id']), None)
        if previous is None or model_hash(previous['fitted_model']) != model_hash(state['current_model']):
            raise ValueError('Source selected model disagrees with the accepted path')
    elif model_hash(state['current_model']) != model_hash(m0):
        raise ValueError('Source depth-zero model is not M0')
    return deepcopy(m0), deepcopy(rows)


def _validate_row(parent, row, config):
    """Regenerate candidate topology, free scope, descriptors, and PASS gates."""
    tid = row['trial_id']
    initialized, specs, features = merge_pair(parent, row['left'], row['right'])
    _same(row['features'], features, tid + '.prefit_features')
    if row.get('n_free') != len(specs) or row.get('n_free_RC') != sum(s['field'] != 'alpha_fraction' for s in specs) or row.get('n_free_alpha') != sum(s['field'] == 'alpha_fraction' for s in specs):
        raise ValueError(f'{tid}: free-parameter counts disagree')
    fitted = row['fitted_model']
    validate_model(fitted)
    if row.get('parent_label') != counts(parent)['label'] or row.get('child_label') != counts(fitted)['label']:
        raise ValueError(f'{tid}: model labels disagree')
    blocks = {b['id']: b for b in fitted['blocks']}
    coordinates = []
    for spec in specs:
        block, field = blocks[spec['block_id']], spec['field']
        if field == 'alpha_fraction':
            value = float(block['alpha'][spec['index']])
            budget = float(spec['budget'])
            if not 0 < value < budget:
                raise ValueError(f'{tid}: fitted alpha is outside its declared local budget')
            coordinates.append(math.log(value) - math.log(budget - value))
        else:
            value = block[field] if 'index' not in spec else block[field][spec['index']]
            coordinates.append(math.log(float(value)))
    rebuilt = apply_parameters(initialized, specs, coordinates)
    _same(fitted, rebuilt, tid + '.conditional_model')
    _same(row['parameter_changes'], parameter_changes(parent, fitted, row['left'], row['right']), tid + '.postfit_changes')
    diagnostic = row.get('diagnostics', {})
    sampler = config['sampler']
    for coordinate in ('transformed', 'physical'):
        d = diagnostic.get(coordinate, {})
        parameters = d.get('per_parameter', [])
        if len(parameters) != len(specs):
            raise ValueError(f'{tid}: incomplete {coordinate} diagnostics')
        required = ('rhat', 'bulk_ess', 'tail_ess')
        if any(not isinstance(p.get(k), (int, float)) or not math.isfinite(p[k]) for p in parameters for k in required):
            # Nonfinite diagnostics are valid unresolved records, never PASS.
            if row.get('status') != 'unresolved':
                raise ValueError(f'{tid}: nonfinite resolved diagnostics')
            continue
        passed = all(p['rhat'] < sampler['rhat_threshold'] and p['bulk_ess'] >= sampler['bulk_ess_min'] and p['tail_ess'] >= sampler['tail_ess_min'] for p in parameters)
        if bool(d.get('passed')) != passed:
            raise ValueError(f'{tid}: {coordinate} diagnostic pass flag disagrees with values')
        _same(d['max_rhat'], max(p['rhat'] for p in parameters), tid + '.max_rhat')
        _same(d['min_bulk_ess'], min(p['bulk_ess'] for p in parameters), tid + '.min_bulk_ess')
        _same(d['min_tail_ess'], min(p['tail_ess'] for p in parameters), tid + '.min_tail_ess')
    converged = bool(diagnostic.get('transformed', {}).get('passed')) and bool(diagnostic.get('physical', {}).get('passed'))
    if bool(diagnostic.get('passed')) != converged or (row.get('sampler_status') == 'converged') != converged:
        raise ValueError(f'{tid}: convergence flags disagree')
    rmse = row.get('metrics', {}).get('validation', {}).get('rmse')
    if not isinstance(rmse, (int, float)) or not math.isfinite(rmse) or rmse < 0:
        raise ValueError(f'{tid}: invalid stored validation RMSE')
    expected = ('pass' if rmse <= config['validation_rmse_limit_C'] else 'fail') if converged else 'unresolved'
    if row.get('status') != expected:
        raise ValueError(f'{tid}: outcome disagrees with unchanged convergence/RMSE gates')


def reprocess_completed(root, config, source, output):
    """Re-select a complete stored path without importing or invoking the sampler.

    The destination must be new/empty. A different replay schema deliberately
    prevents this audit output from being resumed as a production checkpoint.
    """
    from .workflow import (atomic_json, read_json, sha, stable_hash, identity,
                           initial_model, archive, write_csv, discover_all_rules, verify)
    root, output = Path(root).resolve(), Path(output).resolve()
    source_path, source_kind, payload, source_hashes = _source_payload(source)
    if config.get('mode') != 'production':
        raise ValueError('Replay is available only with the production configuration')
    if output == source_path or (source_kind == 'directory' and (output.is_relative_to(source_path) or source_path.is_relative_to(output))):
        raise ValueError('Replay output must be separate from the read-only source')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Replay output already contains files. Choose a new --output directory.')
    started = time.time()
    parent, source_rows = _validate_source(root, config, payload)
    original_state = payload['workflow_state.json']
    by_id = {row['trial_id']: row for row in source_rows}
    used, rows, path, models, selected_checks = set(), [], [], [(0, deepcopy(parent))], []
    data = load_data(root / 'inputs/data.npz')
    for depth in range(int(config['max_stages'])):
        pairs = candidate_pairs(parent)
        if not pairs:
            stop_reason = 'no_component_pair'
            break
        parent_sha = model_hash(parent)
        expected = {f'k{depth:02d}_' + hashlib.sha256(f'{parent_sha}:{left}:{right}'.encode()).hexdigest()[:12]: (left, right)
                    for left, right in pairs}
        available = {r['trial_id'] for r in source_rows if r.get('depth') == depth and r.get('parent_hash') == parent_sha}
        if available != set(expected):
            raise ValueError(f'No complete stored wave for replay parent M{depth}. Replay cannot invent unsampled candidates; use a fresh production run.')
        wave = []
        for tid, (left, right) in expected.items():
            row = deepcopy(by_id[tid])
            if row['left'] != left or row['right'] != right:
                raise ValueError(f'{tid}: stored pair disagrees with regenerated candidate')
            _validate_row(parent, row, config)
            row['source_decision'] = {key: deepcopy(row.get(key)) for key in ('winner', 'pareto_rank', 'comparison')}
            row['source_artifact_directory'] = row.pop('artifact_directory', None)
            row['artifact_directory'] = None
            for attempt in row.get('attempts', []):
                attempt['source_artifact_directory'] = attempt.pop('artifact_directory', None)
                attempt['artifact_directory'] = None
            row['replay_note'] = 'Fit and diagnostics reused; source posterior arrays are not copied or recomputed.'
            wave.append(row)
        wave = rank_trials(wave, min_pass_candidates=1)
        rows.extend(wave); used.update(expected)
        passing = [r for r in wave if r['status'] == 'pass']
        if not passing:
            # No new rescue can be run in a no-MCMC replay. Require the original
            # run to have completed the same zero-PASS terminal parent already.
            if depth != original_state['depth'] or not original_state.get('stop_reason', '').startswith('no_passing_trial'):
                raise ValueError('Zero-PASS replay lacks a completed source terminal decision/rescue policy')
            stop_reason = 'no_passing_trial_with_unresolved' if any(r['status'] == 'unresolved' for r in wave) else 'no_passing_trial'
            break
        winner = next(r for r in wave if r['winner'])
        if depth < original_state['depth'] and original_state['path'][depth]['trial_id'] != winner['trial_id']:
            raise ValueError('Earlier source path changed; stored descendants cannot be reused')
        prediction = simulate(winner['fitted_model'], data)
        metrics = prediction_metrics(prediction, data)
        for partition in ('train', 'validation'):
            _same(metrics[partition]['rmse'], winner['metrics'][partition]['rmse'], winner['trial_id'] + '.' + partition + '_RMSE')
        selected_checks.append(dict(trial_id=winner['trial_id'], model=winner['child_label'],
                                    metrics=metrics, source_diagnostics=deepcopy(winner['diagnostics']),
                                    diagnostic_origin='Original stored MCMC; chains not re-diagnosed'))
        path.append(dict(parent_depth=depth, trial_id=winner['trial_id'], model=winner['child_label'],
                         validation_rmse_C=metrics['validation']['rmse'],
                         d_R=winner['parameter_changes']['d_R'], d_C=winner['parameter_changes']['d_C'],
                         pareto_rank=winner['pareto_rank'], passing_candidates=len(passing),
                         evaluated_candidates=len(wave), selected_from_multiple_pass=len(passing) > 1,
                         selection_reason='sole_passing_candidate' if len(passing) == 1 else 'pareto_among_passing_candidates',
                         unresolved_alternatives=sum(r['status'] == 'unresolved' for r in wave)))
        parent = deepcopy(winner['fitted_model']); models.append((depth + 1, deepcopy(parent)))
    else:
        stop_reason = 'no_component_pair' if not candidate_pairs(parent) else 'configured_max_stages_reached'
    if used != set(by_id):
        raise ValueError('Source contains unvisited candidate rows; replay would silently change the rule population')
    # Do not create a destination until source consistency checks have succeeded.
    output.mkdir(parents=True, exist_ok=True)
    verification = verify(root, output / 'preflight.json')
    provenance = dict(schema='hibo_completed_v2_replay_provenance_v1', source=str(source_path),
                      source_kind=source_kind, source_json_sha256=source_hashes,
                      source_run_hash=original_state['run_hash'], source_status=original_state['status'],
                      source_stop_reason=original_state.get('stop_reason'), source_accepted_steps=original_state['depth'],
                      source_selected_model=counts(original_state['current_model']),
                      source_elapsed_seconds=payload['summary.json'].get('elapsed_seconds'),
                      original_path=deepcopy(original_state['path']), new_mcmc_attempts=0,
                      posterior_arrays_included=False, diagnostics_recomputed=False,
                      selected_predictions_recomputed=True, all_frozen_parent_waves_checked=True,
                      note='Completed fits retained. Only adoption policy and subgroup analysis changed. This is an audit replay, not a resumed MCMC run.')
    for name in SOURCE_FILES:
        atomic_json(output / 'source_snapshot' / name, payload[name])
    for depth, model in models:
        atomic_json(output / f'models/M_{depth:02d}_{counts(model)["label"]}.json', model)
    archive(output, rows)
    atomic_json(output / 'selected_path.json', path)
    atomic_json(output / 'selected_model.json', parent)
    atomic_json(output / 'selected_model_metrics.json', selected_checks[-1]['metrics'] if selected_checks else prediction_metrics(simulate(parent, data), data))
    atomic_json(output / 'selected_prediction_checks.json', selected_checks)
    atomic_json(output / 'provenance.json', provenance)
    rule_result = discover_all_rules(rows, config, output / 'rules')
    state = dict(schema='hibo-audit-replay-v3', run_hash=stable_hash(identity(root, config)),
                 mode='audit_replay', status='completed', started=started, updated=time.time(),
                 depth=len(path), current_model=parent, path=path, rows=rows, stop_reason=stop_reason,
                 verification=verification, provenance=provenance, resumable_mcmc=False)
    atomic_json(output / 'workflow_state.json', state)
    atomic_json(output / 'run_identity.json', identity(root, config))
    summary = dict(status='completed', stop_reason=stop_reason, mode='audit_replay',
                   n_trials=len(rows), n_attempts=sum(len(r.get('attempts', [])) for r in rows),
                   new_mcmc_attempts=0, n_pass=sum(r['status'] == 'pass' for r in rows),
                   n_fail=sum(r['status'] == 'fail' for r in rows), n_unresolved=sum(r['status'] == 'unresolved' for r in rows),
                   current_model=counts(parent), accepted_steps=len(path), minimum_pass_candidates=1,
                   selected_validation_rmse_C=selected_checks[-1]['metrics']['validation']['rmse'] if selected_checks else None,
                   elapsed_seconds=time.time() - started, rule_discovery=rule_result,
                   source_elapsed_seconds=provenance['source_elapsed_seconds'],
                   source_accepted_steps=original_state['depth'], source_current_model=counts(original_state['current_model']),
                   scope='Read-only reuse of completed v2 conditional fits. No new MCMC or external rule validation.',
                   resumable_mcmc=False, posterior_arrays_included=False)
    atomic_json(output / 'summary.json', summary)
    print(f'REPROCESSED: {len(rows)} stored trials, {len(path)} adoptions, {counts(parent)["label"]}; no new MCMC.', flush=True)
    print(f'Results folder: {output}', flush=True)
    return {**summary, "output": str(output)}
