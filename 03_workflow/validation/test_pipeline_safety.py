"""Archive durability and handoff-order regression tests; no MCMC is run."""
from pathlib import Path
import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pipeline_archive_test", PACKAGE / "run_pipeline.py")
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineSafety(unittest.TestCase):
    def test_zip_fsync_uses_writable_descriptor_and_validates_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "source.txt"
            source.write_text("complete initial diagnostics", encoding="utf-8")
            real_fsync = os.fsync
            descriptors_checked = []

            def require_writable(fd):
                # An empty write also fails for read-only descriptors on POSIX.
                # This catches the Windows read-only _commit failure on Linux.
                os.write(fd, b"")
                descriptors_checked.append(fd)
                return real_fsync(fd)

            with patch.object(pipeline.os, "fsync", side_effect=require_writable):
                receipt = pipeline.zipped(folder / "results.zip", [(source, "initial/source.txt")])
            self.assertTrue(descriptors_checked)
            self.assertTrue(receipt["crc_verified"])
            self.assertEqual(receipt["sha256"], pipeline.sha(folder / "results.zip"))
            with zipfile.ZipFile(folder / "results.zip") as archive:
                self.assertEqual(archive.read("initial/source.txt"), source.read_bytes())
            self.assertFalse((folder / "results.zip.tmp").exists())

    def test_crc_failure_preserves_previous_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "source.txt"
            source.write_text("old result", encoding="utf-8")
            destination = folder / "results.zip"
            pipeline.zipped(destination, [(source, "result.txt")])
            original = destination.read_bytes()
            source.write_text("new result", encoding="utf-8")
            with patch.object(pipeline.zipfile.ZipFile, "testzip", return_value="result.txt"):
                with self.assertRaisesRegex(RuntimeError, "ZIP verification failed"):
                    pipeline.zipped(destination, [(source, "result.txt")])
            self.assertEqual(destination.read_bytes(), original)

    def test_missing_required_input_cannot_create_incomplete_success_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            destination = folder / "results.zip"
            with self.assertRaisesRegex(FileNotFoundError, "Required archive input is missing"):
                pipeline.zipped(destination, [(folder / "missing-initial.zip", "Initial_MCMC_results.zip")])
            self.assertFalse(destination.exists())

    def run_orchestration(self, gate_passed):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "package"
            root.mkdir()
            work = Path(temp) / "work"
            initial = work / "initial"
            reduction = work / "reduction"
            output = work / "reduction_results"
            for path in (initial / "results/single_sp_initial", reduction, root / "initial"):
                path.mkdir(parents=True)
            (initial / "results/single_sp_initial/status.json").write_text(json.dumps({
                "stage": "complete" if gate_passed else "settling_gate_failed"}))
            (root / "initial/PACKAGE_MANIFEST_SHA256.json").write_text("{}")
            state = {"workdir": str(work), "stage": "initial_running"}
            events = []

            def archive_initial(s):
                events.append("archive_initial")
                s["initial_archive"] = {"crc_verified": True, "path": "verified.zip"}
                s["stage"] = "initial_archived"
                return s["initial_archive"]

            def gate(_):
                events.append("gate")
                return {"passed": gate_passed, "reasons": [] if gate_passed else ["not converged"]}

            def handoff(*_):
                events.append("handoff")

            def child(_, script, *args):
                events.append("child_" + args[0])
                if args[0] == "run":
                    (output / "rules").mkdir(parents=True)
                    (output / "workflow_state.json").write_text(json.dumps({"status": "completed", "depth": 0}))
                    (output / "rules/summary.json").write_text(json.dumps({"status": "completed"}))
                return 0

            with patch.object(pipeline, "ROOT", root), patch.object(pipeline, "STATE", root / "pipeline_state.json"), \
                 patch.object(pipeline, "STOP", root / "STOP"), \
                 patch.object(pipeline, "archive_initial", side_effect=archive_initial), \
                 patch.object(pipeline, "initial_gate", side_effect=gate), \
                 patch.object(pipeline, "prepare_handoff", side_effect=handoff), \
                 patch.object(pipeline, "child", side_effect=child), \
                 patch.object(pipeline, "archive_reduction", side_effect=lambda _: events.append("archive_reduction")):
                pipeline.proceed(state)
            return events, state

    def test_nonconverged_initial_is_archived_and_blocks_reduction(self):
        events, state = self.run_orchestration(False)
        self.assertEqual(events, ["archive_initial", "gate"])
        self.assertEqual(state["stage"], "initial_not_converged")

    def test_archive_and_gate_precede_all_reduction_operations(self):
        events, state = self.run_orchestration(True)
        self.assertEqual(events, ["archive_initial", "gate", "handoff", "child_verify", "child_run", "archive_reduction"])
        self.assertEqual(state["stage"], "completed")

    def test_start_refuses_existing_run_before_workdir_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            state_path = Path(temp) / "pipeline_state.json"
            state_path.write_text("{}")
            work = Path(temp) / "do_not_create"
            with patch.object(pipeline, "STATE", state_path):
                with self.assertRaisesRegex(RuntimeError, "already has a run"):
                    pipeline.start(work)
            self.assertFalse(work.exists())

    def test_settling_failure_reports_actual_health_not_missing_posterior_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            initial=Path(temp)/'initial'
            folder=initial/'results/single_sp_initial';folder.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps({'stage':'settling_gate_failed','retained_per_chain':0}))
            blockers=['Chain 1: more than 1% invalid or |energy error|>1000 HMC transitions',
                      'Chain 4: more than 1% invalid or |energy error|>1000 HMC transitions']
            values={'max_rhat':1.00718,'min_bulk_ess':812.34,'min_tail_ess':845.01,'invalid_parameters':0}
            (folder/'settling_diagnostics.json').write_text(json.dumps({
                'completed':20000,'physical':values,'ordered':dict(values,max_rhat=1.0113,min_tail_ess=83.79),
                'blocking_failures':blockers,'chain_health':[]}))
            gate=pipeline.initial_gate(initial)
            self.assertFalse(gate['passed'])
            self.assertEqual(gate['production_diagnostics_status'],'not_available_no_retained_posterior')
            self.assertEqual(gate['last_settling_diagnostics']['physical'],values)
            self.assertEqual(gate['last_settling_diagnostics']['blocking_failures'],blockers)
            messages='; '.join(gate['reasons'])
            self.assertIn('Chain 1:',messages)
            self.assertNotIn('max_rhat fails',messages)
            self.assertNotIn('min_tail_ess fails',messages)
            self.assertNotIn('missing retained checkpoint',messages)

    def test_completed_sampler_with_missing_export_is_explicitly_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            initial=Path(temp)/'initial';folder=initial/'results/single_sp_initial';folder.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps({'stage':'complete'}))
            gate=pipeline.initial_gate(initial)
            self.assertFalse(gate['passed'])
            self.assertEqual(gate['production_diagnostics_status'],'export_summary_missing')
            self.assertEqual(len(gate['reasons']),1)
            self.assertNotIn('Rhat','; '.join(gate['reasons']))

    def test_completed_export_still_requires_real_production_thresholds(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            initial=Path(temp)/'initial';folder=initial/'results/single_sp_initial'
            export=folder/'export';export.mkdir(parents=True)
            (folder/'status.json').write_text(json.dumps({'stage':'complete'}))
            (folder/'result_bundle.json').write_text(json.dumps({'posterior_export':{'status':'export_complete'}}))
            values={'max_rhat':1.01,'min_bulk_ess':500.,'min_tail_ess':500.,'invalid_parameters':0,'passed':True}
            summary={'status':'converged_at_first_N','N':6000,'independent_parameters':55,'chains':4,
                     'diagnostics':{'physical':values.copy(),'ordered':values.copy()},
                     'numerical_health':{'passed':True,'blocking_failures':[]}}
            for name in ['parameter_summary.csv','temperatures_1min.csv','raw_joint_chains.npz']:
                (export/name).write_bytes(b'fixture')
            np.savez(folder/'checkpoint.npz',checkpoint_state_json=np.asarray(json.dumps({'N':6000})))
            def save_summary():
                (export/'summary.json').write_text(json.dumps(summary))
                (export/'EXPORT_MANIFEST_SHA256.json').write_text(json.dumps({n:pipeline.sha(export/n) for n in
                    ['summary.json','parameter_summary.csv','temperatures_1min.csv']}))
            save_summary()
            self.assertTrue(pipeline.initial_gate(initial)['passed'])
            summary['numerical_health']={'passed':False,'blocking_failures':['recent window, chain 4: no accepted ordinary HMC transition']}
            save_summary()
            gate=pipeline.initial_gate(initial)
            self.assertFalse(gate['passed'])
            self.assertIn('recent window, chain 4: no accepted ordinary HMC transition',gate['reasons'])
            del summary['numerical_health']
            save_summary()
            gate=pipeline.initial_gate(initial)
            self.assertFalse(gate['passed'])
            self.assertIn('Production numerical health is missing from the independently recomputed export',gate['reasons'])
            summary['numerical_health']={'passed':True,'blocking_failures':[]}
            summary['diagnostics']['physical']['min_bulk_ess']=100.
            save_summary()
            gate=pipeline.initial_gate(initial)
            self.assertFalse(gate['passed'])
            self.assertIn('physical min_bulk_ess fails: 100.0',gate['reasons'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
