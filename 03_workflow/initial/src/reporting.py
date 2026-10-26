"""Shared chain diagnostics and checkpoint summaries."""
import json
import numpy as np
from model import metrics
from diagnostic_parallel import diagnostic_matrix, diagnostic_definition

def save(path, **arrays):
    tmp = path.with_name(path.stem + '.tmp.npz')
    # Store checkpoint arrays without compression; compress final archives once.
    (np.savez if path.name == "checkpoint.npz" else np.savez_compressed)(tmp, **arrays)
    tmp.replace(path)

def diagnostics(m, u):
    p = m.values(u)
    r,b,t=diagnostic_matrix(p)
    valid=np.isfinite(r)&np.isfinite(b)&np.isfinite(t)
    number=lambda x:float(x) if np.isfinite(x) else None
    return dict(diagnostic_definition=diagnostic_definition(), max_rhat=number(r.max()),min_bulk_ess=number(b.min()),min_tail_ess=number(t.min()),
                invalid_parameters=int((~valid).sum()),
                passed=bool(valid.all() and r.max()<1.10 and b.min()>=400 and t.min()>=400),
                per_parameter=[dict(name=n,rhat=number(rr),ess_bulk=number(bb),ess_tail=number(tt)) for n,rr,bb,tt in zip(m.names,r,b,t)])

def canonical_diagnostics(v):
    """Check ordered coordinates as well as randomly restored physical labels."""
    r,b,t=diagnostic_matrix(v)
    valid=np.isfinite(r)&np.isfinite(b)&np.isfinite(t)
    number=lambda x:float(x) if np.isfinite(x) else None
    return dict(diagnostic_definition=diagnostic_definition(), max_rhat=number(r.max()),min_bulk_ess=number(b.min()),min_tail_ess=number(t.min()),
                invalid_parameters=int((~valid).sum()),
                passed=bool(valid.all() and r.max()<1.10 and b.min()>=400 and t.min()>=400))

def summarize(m, u, lp, out, label, diag):
    p = m.native(u)
    flat = p.reshape(-1, m.ndim)
    med = np.median(flat, axis=0)
    v = m.values(u)
    vf = v.reshape(-1, len(m.names))
    q = np.quantile(vf, [0.025, 0.5, 0.975], axis=0)
    pred = m.simulate_native(med)
    baseline = m.simulate_native(np.r_[np.log(m.ref), np.log(0.3), 1.5, np.zeros(28)])
    d = m.d
    save(out / (label + '.npz'), chains_u=u, chains_native=p, median_native=med, parameter_quantiles=q, prediction=pred, baseline=baseline, names=np.array(m.names), logp=lp)
    rows = [dict(parameter=name, reference=float(m.ref[j]) if j < 25 else None, median=float(q[1, j]), p025=float(q[0, j]), p975=float(q[2, j]), **{k: v for k, v in diag['per_parameter'][j].items() if k != 'name'}) for j, name in enumerate(m.names)]
    result = dict(period=m.period, checkpoint=label, chains=4, retained_per_chain=int(u.shape[1]), n_R=17, n_C=8, n_error=2, n_parameters=55, n_reported=59, n_alpha=32, n_alpha_independent=28, train=metrics(pred, d['y'], d['train']), validation=metrics(pred, d['y'], d['valid']), baseline={key: metrics(baseline, d['y'], d[mask]) for key, mask in [('train', 'train'), ('validation', 'valid')]}, likelihood_points=len(m.idx), AR1_segments=m.nblocks, diagnostics=diag, parameters=rows)
    (out / (label + '.json')).write_text(json.dumps(result, indent=2))
    return result
