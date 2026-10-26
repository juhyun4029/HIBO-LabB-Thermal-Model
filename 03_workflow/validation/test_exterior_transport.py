"""Tests of proposal measure, detailed balance, rejection, and persistence.

These exact checks validate the transition kernel rather than convergence
of the building-model posterior."""
from pathlib import Path
import itertools
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "initial" / "src"))
import exterior_transport as target
import multiple_try_heat
from global_transport import from_gap, to_gap, GAPS


def example_q():
    q = np.random.default_rng(155).normal(size=55)
    q[[18, 19, 20, 21, 22, 24]] = [2, 8, 11, 3, 7, 12]
    return q


def numerical_jacobian(fn, x):
    eps = 1e-6
    answer = np.empty((len(fn(x)), len(x)))
    for j in range(len(x)):
        step = np.zeros_like(x)
        step[j] = eps
        answer[:, j] = (fn(x + step) - fn(x - step)) / (2 * eps)
    return answer


class FixedProposal:
    def __init__(self, values, correction=0.567, pair=(0, 1)):
        self.values, self.correction, self.pair = values, correction, pair

    def propose(self, x, rng):
        return self.values.copy(), self.correction, *self.pair


class ExteriorTransportTests(unittest.TestCase):
    def test_unit_coordinate_jacobian_and_unchanged_coordinates(self):
        q = example_q()
        indices = np.array([18, 3, 24])
        expected = np.array([q[18], q[3], q[24] - logsumexp(np.r_[0., q[[1, 3, 9, 13]]])])
        np.testing.assert_allclose(target.coordinates(q), expected, atol=1e-13)
        values = expected + np.array([1., -.2, .15])
        out = target.embed(q, values)
        np.testing.assert_allclose(target.coordinates(out), values, atol=1e-13)
        other = np.setdiff1d(np.arange(55), indices)
        np.testing.assert_array_equal(out[other], q[other])
        def chart(v):
            full = q.copy()
            full[indices] = v
            return target.coordinates(full)
        jac = numerical_jacobian(chart, q[indices])
        self.assertAlmostEqual(np.linalg.det(jac), 1., places=7)

    def test_induced_gap_jacobian_and_target_cancellation(self):
        q = example_q()
        original = target.coordinates(q)
        destination = original + np.array([1., -.2, .15])
        affine = np.array([[.7, .1, .02], [.03, 1.2, -.04], [.01, .08, .9]])
        def move(full):
            return target.embed(full, destination + affine @ (target.coordinates(full) - original))
        v = to_gap(q)
        vnew = to_gap(move(q))
        # Thermal coordinates alone suffice; the other 30 are held fixed.
        def gap_move(thermal):
            full = v.copy()
            full[:25] = thermal
            return to_gap(move(from_gap(full)))[:25]
        jac = numerical_jacobian(gap_move, v[:25])
        expected = np.linalg.slogdet(affine)[1] + sum(v[GAPS] - vnew[GAPS])
        self.assertAlmostEqual(np.linalg.slogdet(jac)[1], expected, places=7)
        cold_difference, mixture_correction = -2.734, .567
        via_gap = cold_difference + sum(vnew[GAPS] - v[GAPS]) + mixture_correction + sum(v[GAPS] - vnew[GAPS])
        self.assertAlmostEqual(via_gap, cold_difference + mixture_correction, places=13)

    def test_actual_transition_passes_correct_measure_to_heat_kernel(self):
        q = example_q()
        values = target.coordinates(q) + np.array([1., -.2, .15])
        mixture = FixedProposal(values)
        candidate = target.embed(q, values)
        heat_info = dict(accepted=1., probability=.5, importance_ess=3., maximum_weight=.5)
        with patch.object(target, 'heat_transition', return_value=(candidate, 12., heat_info)) as heat:
            out, lp, info = target.transition(None, q, 10., np.random.default_rng(1), mixture, 4)
        call = heat.call_args.args
        np.testing.assert_array_equal(call[2], to_gap(candidate)[:25])
        expected = mixture.correction + sum(to_gap(q)[GAPS] - to_gap(candidate)[GAPS])
        self.assertAlmostEqual(call[3], expected, places=13)
        self.assertEqual(call[-1], 4)
        self.assertEqual(info['accepted'], 1.)
        np.testing.assert_array_equal(out, candidate)

    def test_invalid_order_is_single_rejection_and_identity_is_identity(self):
        q = example_q()
        values = target.coordinates(q)
        values[0] = q[19] + 1
        with patch.object(target, 'heat_transition') as heat:
            out, lp, info = target.transition(None, q, 10., None, FixedProposal(values), 4)
            heat.assert_not_called()
        np.testing.assert_array_equal(out, q)
        self.assertEqual(lp, 10.)
        self.assertEqual(info['valid'], 0)
        with patch.object(target, 'heat_transition') as heat:
            out, lp, info = target.transition(None, q, 10., None,
                                               FixedProposal(target.coordinates(q), pair=(1, 1)), 4)
            heat.assert_not_called()
        self.assertEqual(info['identity'], 1)

    def test_fit_uses_bounded_window_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(919)
        records = np.repeat(example_q()[None, None, :], 160, axis=0)
        for t in range(160):
            values = np.array([2. if t % 2 else 6., -.2, 10.]) + rng.normal(size=3) * [.1, .2, .1]
            records[t, 0] = target.embed(records[t, 0], values)
        records[:20] = np.nan  # Excluded prefix must never enter the fit.
        mixture, info = target.fit(records, window=128, min_count=32)
        self.assertTrue(info['enabled'])
        self.assertEqual(info['total_positions_used'], 128)
        self.assertEqual(info['region_counts'], [64, 64])
        arrays = target.to_arrays(mixture)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frozen.npz'
            np.savez(path, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                restored = target.from_arrays(saved)
        for name in ('means', 'covariances', 'weights'):
            np.testing.assert_array_equal(getattr(mixture, name), getattr(restored, name))
        x = target.coordinates(records[-1, 0])
        a = mixture.propose(x, np.random.default_rng(49))
        b = restored.propose(x, np.random.default_rng(49))
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1:], b[1:])
        self.assertIsNone(target.from_arrays(target.to_arrays(None)))
        broken = dict(arrays)
        broken['transport_weights'] = np.array([.4, .4])
        with self.assertRaises(ValueError):
            target.from_arrays(broken)

    def test_exact_finite_state_multiple_try_balance(self):
        # Exhaustive reference for the anchor/bank/accept/reselect algorithm.
        # No Monte Carlo tolerance or implementation-derived target is used.
        pi = np.array([[.08, .22], [.50, .20]])
        alpha_proposal = np.array([[.8, .2], [.3, .7]])
        thermal_proposal = np.array([[.3, .7], [.6, .4]])
        matrix = np.zeros((4, 4))
        for theta, alpha in itertools.product(range(2), repeat=2):
            row = 2 * theta + alpha
            for anchor, other in itertools.product(range(2), repeat=2):
                old = np.zeros(2, dtype=int)
                old[anchor], old[1-anchor] = alpha, other
                aug = .5 * alpha_proposal[theta, other]
                old_w = pi[theta, old] / alpha_proposal[theta, old]
                for new_theta in range(2):
                    for new_bank in itertools.product(range(2), repeat=2):
                        bank = np.array(new_bank)
                        new_w = pi[new_theta, bank] / alpha_proposal[new_theta, bank]
                        proposed = thermal_proposal[theta, new_theta] * np.prod(alpha_proposal[new_theta, bank])
                        ratio = new_w.sum() / old_w.sum() * thermal_proposal[new_theta, theta] / thermal_proposal[theta, new_theta]
                        acceptance = min(1., ratio)
                        for j in range(2):
                            matrix[row, 2*new_theta+bank[j]] += aug*proposed*acceptance*new_w[j]/new_w.sum()
                            matrix[row, 2*theta+old[j]] += aug*proposed*(1-acceptance)*old_w[j]/old_w.sum()
        np.testing.assert_allclose(matrix.sum(1), 1., atol=1e-13)
        weights = pi.ravel()
        np.testing.assert_allclose(weights @ matrix, weights, atol=1e-13)
        flux = weights[:, None] * matrix
        np.testing.assert_allclose(flux, flux.T, atol=1e-13)

    def test_nan_candidate_multiple_try_is_rejected(self):
        class DummyModel:
            latent_dimension = 37
            mu0 = np.full(28, .125)
            calls = 0
            def __init__(self):
                self.base = self.model = self
            def to_raw(self, q): return q
            def alpha(self, q): return np.full((4, 8), .125)
            def estimate(self, theta, noise, latent):
                self.calls += 1
                if self.calls == 2: return np.nan, None
                return 0., dict(alphas=np.full((len(latent), 4, 8), .125),
                                logweights=np.zeros(len(latent)), g={})
            def likelihood(self, a, g): return 0.
            def logq_alpha(self, a, g): return 0.
            def prior_a(self, theta, noise, a): return 0.
            def q_from(self, theta, noise, a): return example_q()
            def logpost(self, q): return 0.
        q = example_q()
        out, lp, info = multiple_try_heat.transition(DummyModel(), q, to_gap(q)[:25],
                                                   0., np.random.default_rng(77), 2)
        self.assertEqual(info['accepted'], 0.)
        self.assertEqual(info['probability'], 0.)
        self.assertTrue(np.isfinite(out).all() and np.isfinite(lp))


if __name__ == '__main__':
    unittest.main()
