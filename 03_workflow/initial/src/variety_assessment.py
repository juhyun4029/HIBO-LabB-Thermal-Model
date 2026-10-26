"""Descriptive temperature-variety assessment after posterior sampling.

Intervals describe parameter uncertainty under the observed EnergyPlus
forcing, including terminal discharge conditions. This postprocessing does
not alter the posterior, stopping criteria, or representative draw."""
from pathlib import Path
import csv
import json
import re
import numpy as np

TEMPERATURE_BINS = ((-np.inf,20.,'<20 C'),(20.,22.,'20 to <22 C'),
                    (22.,24.,'22 to <24 C'),(24.,26.,'24 to <26 C'),
                    (26.,28.,'26 to <28 C'),(28.,np.inf,'>=28 C'))
PHASES = ((0.,5.,'0 to <5 min'),(5.,30.,'5 to <30 min'),
          (30.,120.,'30 to <120 min'),(120.,np.inf,'>=120 min'))
MAX_JOINT_DRAWS = 256


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_joint_indices(shape, maximum=MAX_JOINT_DRAWS):
    """Equal allocation to chains, deterministic midpoints of draw strata."""
    chains, draws = int(shape[0]), int(shape[1])
    limit = min(int(maximum), MAX_JOINT_DRAWS)
    if draws < 1 or chains < 1 or limit < chains:
        raise ValueError('At least one retained draw per chain and a sufficient draw budget are required.')
    per_chain = min(draws, limit // chains)
    positions = np.floor((np.arange(per_chain) + .5) * draws / per_chain).astype(int)
    return [(c, int(j)) for c in range(chains) for j in positions]


def _actual_setpoints(model, source_csv):
    d = model.d
    n = len(d['y'])
    if source_csv is None:
        if 'heating_setpoint' not in d or 'cooling_setpoint' not in d:
            raise ValueError('Actual heating/cooling setpoint arrays are required; no schedule is inferred.')
        heating = np.asarray(d['heating_setpoint'], dtype=float)
        cooling = np.asarray(d['cooling_setpoint'], dtype=float)
        audit = dict(source='processed actual EnergyPlus setpoint arrays; no schedule inferred',
                     source_csv_verified=False)
    else:
        import pandas as pd
        source_csv = Path(source_csv)
        target = model.reference['target_zone']
        required = ['Date/Time', target+':Zone Air Temperature [C](TimeStep)',
                    target+':Zone Thermostat Heating Setpoint Temperature [C](TimeStep)',
                    target+':Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)']
        header = pd.read_csv(source_csv, nrows=0).columns
        lookup = {c.strip().casefold():c for c in header}
        missing = [v for v in required if v.casefold() not in lookup]
        if missing:
            raise ValueError('Required actual EnergyPlus columns missing: '+repr(missing))
        keys = [lookup[v.casefold()] for v in required]
        f = pd.read_csv(source_csv, usecols=keys)
        if len(f) != n:
            raise ValueError('Source CSV and processed model data have different lengths.')
        observed = f[keys[1]].to_numpy(float)
        if not np.allclose(observed, d['y'], rtol=0., atol=1e-9):
            raise ValueError('Source EnergyPlus temperature does not match processed model data.')
        stamp = f[keys[0]].str.extract(r'\s*(\d+)/(\d+)\s+(\d+):(\d+):(\d+)')
        if stamp.isna().any().any():
            raise ValueError('Unrecognized EnergyPlus minute-end timestamp.')
        a = stamp.astype(int).to_numpy()
        minute_end = a[:,2]*60+a[:,3]
        if not (np.array_equal(a[:,0], d['month']) and np.array_equal(a[:,1], d['day'])
                and np.array_equal(minute_end, d.get('minute_end', np.arange(n)%1440+1))
                and np.all(a[:,4] == 0)):
            raise ValueError('Source timestamps and processed month/day/minute endpoints are misaligned.')
        heating = f[keys[2]].to_numpy(float)
        cooling = f[keys[3]].to_numpy(float)
        for key, value in [('heating_setpoint',heating),('cooling_setpoint',cooling)]:
            if key in d and not np.allclose(value,d[key],rtol=0.,atol=1e-12):
                raise ValueError('Source setpoint and processed '+key+' differ.')
        audit = dict(source='actual EnergyPlus heating/cooling columns', source_csv_verified=True,
                     source_csv_name=source_csv.name, columns=keys, aligned_rows=n,
                     maximum_temperature_difference_C=float(np.max(abs(observed-d['y']))))
    if (heating.shape != (n,) or cooling.shape != (n,)
            or not np.isfinite(heating).all() or not np.isfinite(cooling).all()
            or np.any(cooling < heating)):
        raise ValueError('Invalid actual heating/cooling setpoint arrays.')
    return heating, cooling, audit


def _causal_phase(heating, cooling):
    """Age zero labels the first recorded minute with changed commands."""
    n = len(heating)
    changes = np.r_[False, (heating[1:] != heating[:-1]) | (cooling[1:] != cooling[:-1])]
    last = np.maximum.accumulate(np.where(changes,np.arange(n),-1))
    age = np.full(n,np.nan)
    known = last >= 0
    age[known] = np.arange(n)[known]-last[known]
    return changes, age


def _metrics(prediction, observed, mask, score_grid):
    mask = np.asarray(mask, bool)
    n = int(mask.sum())
    if n:
        error = prediction[mask]-observed[mask]
        result = dict(n_minutes=n,rmse_C=float(np.sqrt(np.mean(error**2))),
                      mae_C=float(np.mean(abs(error))),bias_model_minus_EPlus_C=float(error.mean()),
                      p95_absolute_error_C=float(np.quantile(abs(error),.95)),
                      fraction_model_below_EPlus=float(np.mean(error<0)))
    else:
        result = dict(n_minutes=0,rmse_C=None,mae_C=None,bias_model_minus_EPlus_C=None,
                      p95_absolute_error_C=None,fraction_model_below_EPlus=None)
    # Dynamic error uses nonoverlapping five-minute endpoints. Classification is
    # at the current endpoint. The prior endpoint must be in the SAME split,
    # but need not have the same retrospective temperature bin/response phase.
    current = np.flatnonzero(mask & score_grid)
    current = current[current >= 5]
    if len(current):
        dp = prediction[current]-prediction[current-5]
        dy = observed[current]-observed[current-5]
        de = dp-dy
        corr = float(np.corrcoef(dp,dy)[0,1]) if len(dp)>2 and np.std(dp)>0 and np.std(dy)>0 else None
        result.update(n_dynamic_5min=int(len(current)),deltaT_5min_rmse_C=float(np.sqrt(np.mean(de**2))),
                      deltaT_5min_bias_C=float(np.mean(de)),deltaT_5min_correlation=corr)
    else:
        result.update(n_dynamic_5min=0,deltaT_5min_rmse_C=None,deltaT_5min_bias_C=None,
                      deltaT_5min_correlation=None)
    return result


def export_variety_assessment(model, u, out, *, representative_prediction=None,
                              baseline_prediction=None, source_csv=None,
                              max_joint_draws=MAX_JOINT_DRAWS, converged=None):
    """Write descriptive CSV/JSON reports after sampling/export, never a gate."""
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    d = model.d
    y = np.asarray(d['y'],float)
    n = len(y)
    if float(d['dt_seconds']) != 60. or not np.isfinite(y).all():
        raise ValueError('This assessment requires finite one-minute physical data.')
    u = np.asarray(u)
    if u.ndim != 3 or u.shape[-1] != model.ndim:
        raise ValueError('Retained u must preserve chain/draw/parameter axes.')
    indices = _selected_joint_indices(u.shape,max_joint_draws)
    heating,cooling,source_audit = _actual_setpoints(model,source_csv)
    reference = (heating+cooling)/2
    changes,age = _causal_phase(heating,cooling)
    splits = [('Training',np.asarray(d['train'],bool)),('Validation',np.asarray(d['valid'],bool))]
    if any(mask.shape != (n,) for _,mask in splits) or np.any(splits[0][1]&splits[1][1]):
        raise ValueError('Training/validation masks are invalid or overlapping.')
    conditioning = np.asarray(d.get('conditioning',~(splits[0][1]|splits[1][1])),bool)
    if np.any(conditioning & (splits[0][1]|splits[1][1])):
        raise ValueError('Conditioning must not be included in assessment metrics.')
    predictions = np.empty((len(indices),n),float)
    for k,(chain,draw) in enumerate(indices):
        p = np.asarray(model.simulate_native(model.native(u[chain,draw])),float)
        if p.shape != (n,) or not np.isfinite(p).all():
            raise ValueError('Nonfinite/misaligned temperature prediction for retained joint draw '+repr((chain+1,draw+1)))
        predictions[k] = p
    mean = predictions.mean(0)
    lower,upper = np.quantile(predictions,[.025,.975],axis=0)
    models = [('joint_draw_posterior_mean',mean)]
    for label,p in [('highest_log_posterior_joint_draw',representative_prediction),
                    ('IDF_prior_reference',baseline_prediction)]:
        if p is not None:
            p=np.asarray(p,float)
            if p.shape != (n,) or not np.isfinite(p).all():
                raise ValueError('Invalid '+label+' prediction.')
            models.append((label,p))
    grids = {}
    for split,mask in splits:
        grid = (np.arange(n)%5==4)&mask
        grid[:5] = False
        grid[5:] &= mask[:-5]
        grids[split] = grid
    overall=[];bins=[];phases=[];levels=[];bands=[];draw_rows=[];chain_rows=[];hold_rows=[];hold_groups=[]
    minute_end=np.asarray(d.get('minute_end',np.arange(n)%1440+1),int)
    block_starts=np.r_[0,np.flatnonzero(changes)]
    block_ends=np.r_[block_starts[1:],n]
    block_duration=np.empty(n,int)
    for start,end in zip(block_starts,block_ends):
        block_duration[start:end]=end-start
    def band_metrics(mask,label):
        if label!='joint_draw_posterior_mean' or not mask.any():
            return dict(parameter_only_band_fraction_EPlus_inside=None,parameter_only_band_mean_width_C=None)
        return dict(parameter_only_band_fraction_EPlus_inside=float(np.mean((y[mask]>=lower[mask])&(y[mask]<=upper[mask]))),
                    parameter_only_band_mean_width_C=float(np.mean(upper[mask]-lower[mask])))
    for split,mask in splits:
        grid=grids[split]
        for label,p in models:
            overall.append(dict(model=label,split=split,**_metrics(p,y,mask,grid)))
            for lo,hi,text in TEMPERATURE_BINS:
                subset=mask&(y>=lo)&(y<hi)
                bins.append(dict(model=label,split=split,observed_temperature_bin=text,
                                 **_metrics(p,y,subset,grid)))
            phase_masks=[('No observed prior setpoint change',~np.isfinite(age))]
            phase_masks += [(text,(age>=lo)&(age<hi)) for lo,hi,text in PHASES]
            for text,phase_mask in phase_masks:
                phases.append(dict(model=label,split=split,response_phase=text,
                                   **_metrics(p,y,mask&phase_mask,grid)))
            for value in np.unique(reference):
                levels.append(dict(model=label,split=split,commanded_central_setpoint_C=float(value),
                                   **_metrics(p,y,mask&(reference==value),grid)))
            # Hold duration is a retrospective scheduled-block descriptor,
            # distinct from causal elapsed phase. Calendar-edge blocks are censored.
            for block,(start,end) in enumerate(zip(block_starts,block_ends)):
                subset=mask&(np.arange(n)>=start)&(np.arange(n)<end)
                if not subset.any():
                    continue
                hold_rows.append(dict(model=label,split=split,hold_block=block+1,
                    first_record_one_based=int(start+1),last_record_one_based=int(end),
                    first_day=int(d['day'][start]),first_minute_end=int(minute_end[start]),
                    commanded_central_setpoint_C=float(reference[start]),
                    preceding_central_setpoint_C=float(reference[start-1]) if start>0 else None,
                    jump_C=float(reference[start]-reference[start-1]) if start>0 else None,
                    observed_block_duration_minutes=int(end-start),
                    left_calendar_censored=bool(start==0),right_calendar_censored=bool(end==n),
                    scored_block_partial_due_to_split=bool(int(subset.sum())!=end-start),
                    **_metrics(p,y,subset,grid),**band_metrics(subset,label)))
            for duration in np.unique(block_duration):
                subset=mask&(block_duration==duration)
                relevant=[(start,end)for start,end in zip(block_starts,block_ends)
                          if end-start==duration and mask[start:end].any()]
                hold_groups.append(dict(model=label,split=split,observed_block_duration_minutes=int(duration),
                    n_contributing_hold_blocks=len(relevant),
                    n_calendar_censored_blocks=sum(start==0 or end==n for start,end in relevant),
                    n_blocks_partial_due_to_split=sum(int(mask[start:end].sum())!=end-start for start,end in relevant),
                    **_metrics(p,y,subset,grid),**band_metrics(subset,label)))
        inside=(y[mask]>=lower[mask])&(y[mask]<=upper[mask])
        bands.append(dict(split=split,n_minutes=int(mask.sum()),
            mean_parameter_only_95pct_width_C=float(np.mean(upper[mask]-lower[mask])) if mask.any() else None,
            fraction_EPlus_inside_parameter_only_95pct_band=float(inside.mean()) if mask.any() else None,
            warning='Parameter-only latent interval, NOT an observation predictive interval; coverage is descriptive, not a pass gate.'))
        for k,(chain,draw) in enumerate(indices):
            draw_rows.append(dict(chain=chain+1,retained_draw=draw+1,split=split,
                                  **_metrics(predictions[k],y,mask,grid)))
        for chain in range(u.shape[0]):
            selected=[k for k,(c,_) in enumerate(indices) if c==chain]
            cp=predictions[selected].mean(0)
            chain_rows.append(dict(chain=chain+1,selected_joint_draws=len(selected),split=split,
                                   **_metrics(cp,y,mask,grid)))
    _csv(out/'variety_overall_metrics.csv',overall)
    _csv(out/'variety_observed_temperature_bins.csv',bins)
    _csv(out/'variety_setpoint_response_phases.csv',phases)
    _csv(out/'variety_commanded_setpoint_levels.csv',levels)
    _csv(out/'variety_hold_segment_metrics.csv',hold_rows)
    _csv(out/'variety_hold_duration_metrics.csv',hold_groups)
    _csv(out/'variety_joint_draw_metrics.csv',draw_rows)
    _csv(out/'variety_chain_mean_prediction_metrics.csv',chain_rows)
    _csv(out/'variety_parameter_only_band_summary.csv',bands)
    with (out/'variety_parameter_only_temperature_bands_1min.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['month','day','minute_end','split','EnergyPlus_C','commanded_central_setpoint_C',
                         'minutes_since_latest_observed_setpoint_change','joint_draw_mean_C',
                         'parameter_only_p025_C','parameter_only_p975_C'])
        for t in range(n):
            split='Conditioning' if conditioning[t] else 'Training' if splits[0][1][t] else 'Validation' if splits[1][1][t] else 'Unscored'
            writer.writerow([int(d['month'][t]),int(d['day'][t]),int(minute_end[t]),
                             split,float(y[t]),float(reference[t]),float(age[t]) if np.isfinite(age[t]) else '',
                             float(mean[t]),float(lower[t]),float(upper[t])])
    summary=dict(schema='hibo-temperature-variety-assessment-v1',
        purpose='Descriptive assessment of the same fixed17R8C temperature-only effective-parameter posterior; no fitting/selection/stopping changes.',
        numerical_convergence_status='diagnostic_gate_passed' if converged else 'provisional_not_converged' if converged is False else 'not_provided',
        quality_gate='NONE: no new fit-quality threshold or automatic quality pass is introduced.',
        source_setpoint_audit=source_audit,
        selected_joint_draws=len(indices),selection='Deterministic equal allocation to chains; one midpoint per equally sized retained-draw stratum. No validation-based draw selection and no sampler RNG consumption.',
        selected_draw_indices_one_based=[dict(chain=c+1,retained_draw=j+1) for c,j in indices],
        retained_chains=int(u.shape[0]),retained_draws_per_chain=int(u.shape[1]),
        maximum_joint_draw_budget=MAX_JOINT_DRAWS,
        uncertainty='Approximate finite-subset pointwise 2.5/97.5% quantiles across at most256 coherent joint-draw latent trajectories. Parameter uncertainty only; no OU residual or observation noise added; not simultaneous bands. May be unstable with few draws or poor chain mixing.',
        prediction_conditioning='Full continuous physical simulation including initial conditioning; validation states are never reset and validation observations are never assimilated. Future physical inputs are the known EPlus inputs; not an autonomous HVAC forecast.',
        score_intervals=dict(temperature_metrics_minutes=1,dynamic_change_minutes=5,
            deltaT_definition='[Tmodel(t)-Tmodel(t-5min)] - [TEPlus(t)-TEPlus(t-5min)] on nonoverlapping5minendpoints; both endpoints must lie in the same scored split; current endpoint determines the retrospective bin/phase.'),
        stratification=dict(observed_temperature='Retrospective strata based on observed EPlus temperature, not on commanded setpoint; include under20/over28 rather than clipping.',
            commanded_setpoint='Actual(heating+cooling)/2 from EPlus outputs; distinct from actual indoor temperature.',
            response_phase='A change in either actual heating or cooling command starts an event. First recorded minute with changed command is age0min, matching array-index phase convention; that record is a minute-end observation. Uses latest past/current event only, no future event information. Initial plateau has no observed prior step.',
            phase_bounds_minutes=['0<=age<5','5<=age<30','30<=age<120','age>=120','no observed prior change'],
            hold_duration='Observed contiguous command-block length, not time since prior change. Initial/final calendar blocks marked censored; blocks intersecting split boundaries marked partial. Group metrics pool minutes descriptively and report contributing block counts, not independent sample sizes.'),
        split_counts=[dict(split=s,n_minutes=int(mask.sum()),observed_setpoint_changes=int((changes&mask).sum()))for s,mask in splits],
        overall_metrics=overall,parameter_only_band_summary=bands,
        limitations=['Metrics do not establish unique individual R/C/alpha identification or equality to EPlus material properties.',
                     'Broader temperature coverage alone is not a guarantee of informative excitation.',
                     'Retrospective stratum metrics do not isolate causal setpoint effects from weather, gains and HVAC interactions.',
                     'No held-out metric changes posterior, stopping criteria, adaptation, or representative draw selection.',
                     'Observed systematic bias and response-regime errors remain visible; numerical convergence does not certify prediction adequacy.'])
    _json(out/'variety_assessment.json',summary)
    return dict(status='variety_assessment_exported',selected_joint_draws=len(indices),
                numerical_convergence_status=summary['numerical_convergence_status'],
                report='variety_assessment.json')
