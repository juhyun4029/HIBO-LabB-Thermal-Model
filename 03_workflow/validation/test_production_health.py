"""Regression tests for numerical health during retained production/export."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'initial/src'))
from production_health import production_health


def healthy(draws=5000):
    stats = np.zeros((4, draws, 38))
    stats[:, :, 1] = 1  # ordinary HMC, all proposals accepted
    return stats


class ProductionHealthTests(unittest.TestCase):
    def test_exactly_one_percent_passes_but_above_it_fails(self):
        stats = healthy(4000)
        stats[0, :40, 5] = 1
        stats[0, :25, 28] = 1
        stats[0, 25:40, 29] = 1
        report = production_health(stats)
        self.assertTrue(report['passed'])
        row = report['all_retained']['chain_health'][0]
        self.assertEqual(row['invalid_or_abs_energy_over_1000_count'], 40)
        self.assertEqual(row['numerically_invalid_count'], 25)
        self.assertEqual(row['abs_energy_over_1000_count'], 15)
        stats[0, 40, [5, 29]] = 1
        self.assertFalse(production_health(stats)['passed'])

    def test_recent_failure_cannot_be_diluted_by_long_history(self):
        stats = healthy(10000)
        stats[0, -50:, [5, 28]] = 1
        report = production_health(stats)
        self.assertTrue(report['all_retained']['passed'])
        self.assertFalse(report['recent']['passed'])
        self.assertFalse(report['passed'])
        self.assertEqual(report['recent']['first_retained_draw'], 6001)
        self.assertEqual(report['recent']['last_retained_draw'], 10000)

    def test_clean_recent_window_does_not_erase_bad_retained_history(self):
        stats = healthy(10000)
        stats[1, :150, [5, 29]] = 1
        report = production_health(stats)
        self.assertFalse(report['all_retained']['passed'])
        self.assertTrue(report['recent']['passed'])
        self.assertFalse(report['passed'])

    def test_bridge_moves_do_not_dilute_ordinary_hmc_failure_rate(self):
        stats = healthy(4000)
        stats[0, :3900, 0] = 7
        stats[0, 3900:3902, [5, 28]] = 1
        report = production_health(stats)
        self.assertFalse(report['passed'])
        row = report['all_retained']['chain_health'][0]
        self.assertEqual(row['hmc_attempts'], 100)
        self.assertEqual(row['invalid_or_abs_energy_over_1000_fraction'], .02)

    def test_each_chain_needs_attempt_accept_and_finite_flags(self):
        stats = healthy(100)
        stats[0, :, 0] = 8
        stats[1, :, 1] = 0
        stats[2, 10, 28] = np.nan
        report = production_health(stats)
        self.assertFalse(report['passed'])
        failures = '; '.join(report['blocking_failures'])
        self.assertIn('chain 1: no ordinary HMC attempts', failures)
        self.assertIn('chain 2: no accepted ordinary HMC', failures)
        self.assertIn('chain 3: nonfinite HMC health', failures)
        json.dumps(report, allow_nan=False)

    def test_missing_empty_or_wrong_shape_fails_closed(self):
        for stats in (None, np.empty((4, 0, 38)), np.zeros((4, 500, 18))):
            with self.subTest(shape=getattr(stats, 'shape', None)):
                self.assertFalse(production_health(stats)['passed'])

    def test_report_does_not_mutate_statistics(self):
        stats = healthy(5000)
        before = stats.copy()
        self.assertTrue(production_health(stats)['passed'])
        np.testing.assert_array_equal(stats, before)


class ExportHealthTests(unittest.TestCase):
    def test_healthy_checkpoint_exports_its_independently_recomputed_health(self):
        import july_export
        import reporting
        import variety_assessment
        class FixtureModel:
            ref = np.ones(25)
            names = [f'parameter_{i}' for i in range(59)]
            idx = np.arange(2)
            d = {'conditioning':np.array([False, False]),
                 'train':np.array([True, False]), 'valid':np.array([False, True]),
                 'month':np.array([7, 7]), 'day':np.array([2, 23]),
                 'y':np.array([22., 24.])}
            def values(self, u):
                return np.ones((*u.shape[:-1], 59))
            def native(self, u):
                return u
            def simulate_native(self, native):
                return np.array([22., 24.])
        passing = dict(passed=True, invalid_parameters=0, max_rhat=1.,
                       min_bulk_ess=500., min_tail_ess=500., per_parameter=[])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savez(folder/'checkpoint.npz',
                     checkpoint_state_json=np.asarray(json.dumps({'N':500})),
                     u=np.zeros((4, 500, 55)), w=np.zeros((4, 500, 55)),
                     logp=np.zeros((4, 500)), statistics=healthy(500))
            (folder/'configuration.json').write_text('{}')
            (folder/'production_health.json').write_text(json.dumps({'passed':False}))
            with patch.object(july_export, 'Model', return_value=FixtureModel()), \
                 patch.object(july_export, 'checkpoint_metrics', return_value=dict(chain=1, retained_draw=1)), \
                 patch.object(reporting, 'diagnostics', return_value=passing), \
                 patch.object(reporting, 'canonical_diagnostics', return_value=passing), \
                 patch.object(variety_assessment, 'export_variety_assessment'):
                result = july_export.export_run(folder)
            self.assertTrue(result['converged'])
            summary = json.loads((folder/'export/summary.json').read_text())
            self.assertTrue(summary['numerical_health']['passed'])
            self.assertEqual(summary['numerical_health']['all_retained']['draws_per_chain'], 500)
            self.assertIn('independently recomputed', summary['numerical_health']['source'])

    def test_saved_passing_health_cannot_override_actual_checkpoint_failure(self):
        import july_export
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            stats = healthy(500)
            stats[2, :, 1] = 0
            np.savez(folder/'checkpoint.npz',
                     checkpoint_state_json=np.asarray(json.dumps({'N':500})),
                     u=np.zeros((4, 500, 55)), w=np.zeros((4, 500, 55)),
                     logp=np.zeros((4, 500)), statistics=stats)
            (folder/'production_health.json').write_text(json.dumps({'passed':True}))
            with patch.object(july_export, 'Model') as model:
                with self.assertRaisesRegex(AssertionError, 'production numerical health'):
                    july_export.export_run(folder)
                model.assert_not_called()
            summary = json.loads((folder/'export/summary.json').read_text())
            self.assertEqual(summary['status'], 'NOT_CONVERGED_numerical_health_failed')
            self.assertFalse(summary['numerical_health']['passed'])
            self.assertFalse(json.loads((folder/'production_health.json').read_text())['passed'])

    def test_saved_N_without_transition_statistics_is_not_exported_as_converged(self):
        import july_export
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            np.savez(folder/'checkpoint.npz',
                     checkpoint_state_json=np.asarray(json.dumps({'N':500})),
                     u=np.zeros((4, 500, 55)), w=np.zeros((4, 500, 55)),
                     logp=np.zeros((4, 500)))
            with self.assertRaisesRegex(AssertionError, 'production numerical health'):
                july_export.export_run(folder)


if __name__ == '__main__':
    unittest.main(verbosity=2)
