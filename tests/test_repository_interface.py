"""Path isolation and forwarding checks for the repository entry point."""

import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repository_run", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RepositoryInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repository = Path(self.temp.name) / "research repo"
        self.repository.mkdir()

    def test_default_start_uses_unique_run_directory(self):
        first = runner.resolve_workdir(self.repository, None)
        second = runner.resolve_workdir(self.repository, None)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.repository / "runs")
        self.assertFalse(first.exists())

    def test_relative_workdir_is_relative_to_repository(self):
        target = runner.resolve_workdir(self.repository, Path("runs/experiment_1"))
        self.assertEqual(target, self.repository / "runs" / "experiment_1")

    def test_external_directory_is_supported(self):
        external = Path(self.temp.name) / "external runs" / "experiment_1"
        self.assertEqual(runner.resolve_workdir(self.repository, external), external)

    def test_published_results_and_source_paths_are_protected(self):
        for name in ("01_energyplus", "02_thermal_model", "03_workflow", "04_results", "05_analysis"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.resolve_workdir(self.repository, Path(name) / "new_run")

    def test_repository_and_ancestor_are_protected(self):
        for path in (self.repository, self.repository.parent, self.repository / "runs"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.resolve_workdir(self.repository, path)

    def test_dotdot_cannot_escape_into_results(self):
        with self.assertRaises(ValueError):
            runner.resolve_workdir(self.repository, Path("runs/../04_results/new_run"))

    def test_symlink_cannot_redirect_run_into_results(self):
        results = self.repository / "04_results"
        results.mkdir()
        runs = self.repository / "runs"
        runs.mkdir()
        link = runs / "redirect"
        try:
            link.symlink_to(results, target_is_directory=True)
        except OSError:
            self.skipTest("Directory symlinks are unavailable on this platform.")
        with self.assertRaises(ValueError):
            runner.resolve_workdir(self.repository, link / "new_run")

    def test_start_forwards_explicit_absolute_workdir(self):
        args = argparse.Namespace(action="start", workdir=Path("runs/experiment_1"))
        command = runner.build_command(args, self.repository, "python-test")
        self.assertEqual(command, [
            "python-test", "-u", str(self.repository / "03_workflow" / "run_pipeline.py"),
            "start", "--workdir", str(self.repository / "runs" / "experiment_1"),
        ])

    def test_other_actions_forward_without_new_workdir(self):
        for action in ("verify", "resume", "status", "stop", "export"):
            args = argparse.Namespace(action=action, workdir=None)
            with self.subTest(action=action):
                command = runner.build_command(args, self.repository, "python-test")
                self.assertEqual(command[-1], action)
                self.assertNotIn("--workdir", command)


if __name__ == "__main__":
    unittest.main()
