"""Discarded-warmup tuning helpers; no target density is defined here."""
import math
import numpy as np


def regularized_covariance(covariance, shrinkage=0.20, eigen_floor=0.01):
    """Shrink correlations, bound their eigenvalues, retain positive scales.

    The eigenvalue floor is applied in standardized correlation coordinates,
    rather than imposing one absolute scale on differently scaled parameters.
    """
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError('Covariance must be square')
    if (not np.isfinite(cov).all() or not 0 < shrinkage <= 1
            or not np.isfinite(eigen_floor) or eigen_floor <= 0):
        raise ValueError('Invalid covariance or shrinkage')
    cov = (cov + cov.T) / 2
    variances = np.diag(cov).copy()
    if np.any(variances <= 0):
        raise ValueError('Every covariance diagonal must be positive')
    scale = np.sqrt(variances)
    correlation = cov / np.outer(scale, scale)
    correlation = (1 - shrinkage) * correlation + shrinkage * np.eye(len(cov))
    eig, vectors = np.linalg.eigh((correlation + correlation.T) / 2)
    eig = np.maximum(eig, eigen_floor)
    correlation = (vectors * eig) @ vectors.T
    normalizer = np.sqrt(np.diag(correlation))
    correlation /= np.outer(normalizer, normalizer)
    answer = correlation * np.outer(scale, scale)
    np.linalg.cholesky(answer)
    return (answer + answer.T) / 2


def momentum_metric(covariance, shrinkage=0.20):
    """M = C^-1 because p~N(0,M), kinetic=.5 p' M^-1 p."""
    covariance = regularized_covariance(covariance, shrinkage)
    metric = np.linalg.solve(covariance, np.eye(len(covariance)))
    metric = (metric + metric.T) / 2
    np.linalg.cholesky(metric)
    return metric, covariance


def pooled_covariance(records, fallback_covariance=None):
    """Pool discarded tuning draws without interpreting them as posterior."""
    records = np.asarray(records, dtype=float)
    flat = records.reshape(-1, records.shape[-1])
    if len(flat) < 2 or not np.isfinite(flat).all():
        raise ValueError('At least two finite tuning positions are required')
    empirical = np.cov(flat, rowvar=False)
    if fallback_covariance is not None:
        # Retain a modest covariance-scale anchor when warmup is correlated.
        empirical = 0.90 * empirical + 0.10 * np.asarray(fallback_covariance)
    return empirical


class DualAveraging:
    """Nesterov dual averaging of log step size, used only during warmup."""
    def __init__(self, initial_step, target=0.90, gamma=0.05,
                 t0=10.0, kappa=0.75, lower=1e-5, upper=0.5):
        self.target, self.gamma, self.t0, self.kappa = target, gamma, t0, kappa
        self.lower, self.upper = lower, upper
        self.n = 0
        self.hbar = 0.0
        self.mu = math.log(10 * initial_step)
        self.log_step = self.log_average = math.log(initial_step)

    def update(self, acceptance_probability):
        if not np.isfinite(acceptance_probability) or not 0 <= acceptance_probability <= 1:
            raise ValueError('Acceptance probability must be finite and in [0,1]')
        self.n += 1
        eta = 1.0 / (self.n + self.t0)
        self.hbar = ((1 - eta) * self.hbar
                     + eta * (self.target - float(acceptance_probability)))
        raw = self.mu - math.sqrt(self.n) * self.hbar / self.gamma
        self.log_step = float(np.clip(raw, math.log(self.lower), math.log(self.upper)))
        weight = self.n ** (-self.kappa)
        self.log_average = weight * self.log_step + (1 - weight) * self.log_average
        return math.exp(self.log_step)

    def final_step(self):
        return float(np.clip(math.exp(self.log_average), self.lower, self.upper))

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, state):
        obj = cls(math.exp(state['log_step']), state['target'])
        obj.__dict__.update(state)
        return obj


def warmup_windows(total):
    """Increasing covariance windows followed by a step-only final buffer."""
    if total < 1200:
        raise ValueError('Tuned warmup requires at least 1200 discarded iterations')
    # A 3000-iteration warmup updates at 250, 500, 1000, and 2000; M is then fixed.
    endpoints = sorted(set(max(100, int(round(total * fraction)))
                           for fraction in (1/12, 1/6, 1/3, 2/3)))
    return [value for value in endpoints if value < total]
