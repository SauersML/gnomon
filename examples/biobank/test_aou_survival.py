"""Small deterministic tests; run on MSI, never against participant data."""
import json
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import aou_survival as aou
import submit_aou
from aou_identity import check_account
from aou_status import failure_label, publish_status
import disease_selection as selection
import aou_score_transform as transforms
import reference_ctn
from aou_checkpoint import StudyCheckpoint
from aou_evaluation import audit_groups, loss_summary


class SurvivalContractTests(unittest.TestCase):
    def test_external_reference_rejects_wrong_score_projection_and_corruption(self):
        import hashlib
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_bytes = b"frozen external model"
            manifest = {"schema": "external-reference-ctn-v1", "reference_population": "1000G",
                        "pgs_id": "PGS004525", "score_column": "PGS004525_AVG",
                        "pc_columns": ["PC1", "PC2"], "projection_model_sha256": "a" * 64,
                        "score_file_sha256": "b" * 64, "training_table_sha256": "c" * 64,
                        "model_sha256": hashlib.sha256(model_bytes).hexdigest()}
            archive = root / "reference.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for name, payload in [("manifest.json", json.dumps(manifest).encode()),
                                      ("transform.gamfit", model_bytes)]:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(payload)
                    tar.addfile(entry, io.BytesIO(payload))
            engine = Mock()
            with patch.dict(sys.modules, {"gamfit": engine}):
                model, path, restored = reference_ctn.load_reference(
                    [archive], root / "loaded", "PGS004525", 2, "a" * 64)
                self.assertIs(model, engine.load.return_value)
                self.assertEqual(path.read_bytes(), model_bytes)
                engine.fit.assert_not_called()
                for pgs, sha, message in [("PGS004536", "a" * 64, "exactly one"),
                                           ("PGS004525", "d" * 64, "same PC projection")]:
                    with self.assertRaisesRegex(ValueError, message):
                        reference_ctn.load_reference([archive], root / "bad", pgs, 2, sha)
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    reference_ctn.load_reference([archive, archive], root / "bad", "PGS004525", 2, "a" * 64)
                with tarfile.open(archive, "w:gz") as tar:
                    for name, payload in [("manifest.json", json.dumps(manifest).encode()),
                                          ("transform.gamfit", b"corrupt model")]:
                        entry = tarfile.TarInfo(name)
                        entry.size = len(payload)
                        tar.addfile(entry, io.BytesIO(payload))
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    reference_ctn.load_reference([archive], root / "bad", "PGS004525", 2, "a" * 64)
            changed = dict(manifest, reference_population="AoU")
            with self.assertRaisesRegex(ValueError, "external reference population"):
                reference_ctn.validate_manifest(changed, "PGS004525", 2, "a" * 64)

    def test_workspace_diagnostic_reads_only_unfinished_worker_logs(self):
        wdl = Path(__file__).with_name("aou_diagnostic.wdl").read_text()
        code = textwrap.dedent(wdl.split("<<'PY'\n", 1)[1].split("    PY\n", 1)[0])
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stderr = directory / "stderr"
            (directory / "score_files.json").write_text("[]")
            (directory / "score_id.json").write_text('""')
            stderr.write_text("RuntimeError: private task failure\n")
            checkpoint = directory / "checkpoint.tar.gz"
            with tarfile.open(checkpoint, "w:gz") as archive:
                for name, content in {
                    "complete/fit.log": "ModuleNotFoundError: private completed history",
                    "complete/completed.json": "{}",
                    "unfinished/fit.log": "ValueError: private unfinished history",
                    "fit.log": "Could not determine input format for private-input\n> Progress: 5/10 variants (50%)",
                }.items():
                    member = tarfile.TarInfo(name)
                    encoded = content.encode()
                    member.size = len(encoded)
                    archive.addfile(member, io.BytesIO(encoded))
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(sys, "argv", ["diagnose", str(stderr), "", str(checkpoint)]), \
                     patch.dict(sys.modules, {"aou_identity": SimpleNamespace(
                         task_account=lambda: None, require_spot_amd=lambda: None)}):
                    exec(compile(code, "aou_diagnostic.wdl", "exec"), {})
            finally:
                os.chdir(previous)
            labels = {path.name: path.read_text() for path in directory.glob("diagnostic__*.txt")}
            self.assertEqual(labels, {"diagnostic__runtime_error.txt": "runtime_error\n",
                                      "diagnostic__value_error.txt": "value_error\n",
                                      "diagnostic__score_input_format.txt": "score_input_format\n",
                                      "diagnostic__score_progress_50_74.txt": "score_progress_50_74\n"})

    def test_workspace_diagnostic_buckets_solver_progress_without_counts(self):
        wdl = Path(__file__).with_name("aou_diagnostic.wdl").read_text()
        code = textwrap.dedent(wdl.split("<<'PY'\n", 1)[1].split("    PY\n", 1)[0])
        solver_log = ("worker_fit_started\n[warm-start-cache] restored persistent warm start key=k\n"
                      + "[PIRLS/joint-Newton mode certificate] returned beta certified\n" * 12)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "score_files.json").write_text("[]")
            (directory / "score_id.json").write_text('""')
            checkpoint = directory / "checkpoint.tar.gz"
            with tarfile.open(checkpoint, "w:gz") as archive:
                for name, content in {
                    "development/PGS004525/pc_varying_ctn_2/fit.log": solver_log,
                    "development/PGS004525/pc_varying_ctn_2/fit.resources.json": json.dumps({
                        "wall_seconds": 100., "cpu_seconds": 3000., "average_cpu_cores": 30.,
                        "concurrent_fits": 2, "solver_threads": 48, "allotted_threads": 64}),
                }.items():
                    member = tarfile.TarInfo(name)
                    encoded = content.encode()
                    member.size = len(encoded)
                    archive.addfile(member, io.BytesIO(encoded))
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with patch.object(sys, "argv", ["diagnose", "", "", str(checkpoint)]), \
                     patch.dict(sys.modules, {"aou_identity": SimpleNamespace(
                         task_account=lambda: None, require_spot_amd=lambda: None)}):
                    exec(compile(code, "aou_diagnostic.wdl", "exec"), {})
            finally:
                os.chdir(previous)
            labels = {path.name for path in directory.glob("diagnostic__*.txt")}
        self.assertEqual(labels, {"diagnostic__worker_fit_started.txt",
                                  "diagnostic__fit_warm_start_restored.txt",
                                  "diagnostic__fit_inner_solves_10_49.txt",
                                  "diagnostic__fit_cpu_partial.txt"})

    def test_compute_bounds_never_change_the_checkpoint_identity(self):
        from aou_checkpoint import result_identity
        settings = {"num_pcs": 6, "fit_timeout_seconds": 600, "query_timeout_seconds": 120,
                    "maximum_bytes_billed": 10**11, "timeout_seconds": 600, "seed": 7}
        retuned = dict(settings, fit_timeout_seconds=4500, query_timeout_seconds=300,
                       maximum_bytes_billed=10**12, timeout_seconds=900)
        self.assertEqual(result_identity(settings), {"num_pcs": 6, "seed": 7})
        self.assertEqual(result_identity(settings), result_identity(retuned))
        self.assertNotEqual(result_identity(settings), result_identity(dict(settings, num_pcs=8)))

    def test_sparse_censoring_support_is_not_reported_as_valid_accuracy(self):
        train = pd.DataFrame({"event_code": [1, 2] * 30, "followup": [2.] * 60,
                              "ancestry": ["major"] * 59 + ["rare"]})
        test = pd.DataFrame({"event_code": [1, 0] * 10, "followup": [2.] * 20,
                             "ancestry": ["rare"] * 20})
        config = {"min_train_events_per_cause": 30, "min_report_count": 20,
                  "horizons_years": [1.]}
        self.assertEqual(aou.fit_support(train, test, config), [])
        self.assertTrue(aou.partition_support(train, test, config))
        metrics = aou.evaluate(train, test, np.full((20, 1), .1), [1.], 20)
        self.assertEqual(metrics[0]["status"], "insufficient_support")
        self.assertNotIn("brier", metrics[0])
        self.assertTrue(aou.fit_support(train.iloc[:20], test, config))

    def test_failed_worker_retains_private_log_without_completion_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            checkpoint = StudyCheckpoint(directory, "gs://workspace/checkpoint.tar.gz",
                                         "project", "runtime@example.org", {})
            retained = []
            def publish():
                retained.append((directory / "fit.log").read_text())
            with patch.object(checkpoint, "publish", side_effect=publish):
                with self.assertRaisesRegex(RuntimeError, "fit failed"):
                    aou.checkpointed_fit(
                        [sys.executable, "-c", "print('worker failure evidence'); raise SystemExit(1)"],
                        5, directory / "fit.log", checkpoint)
            self.assertEqual(retained, ["worker failure evidence\n"])
            self.assertFalse(checkpoint.step_is_complete(directory, model=True))

    def test_prespecified_score_retains_its_cohort(self):
        disease = {"candidates": ["PGS004525"]}
        first = pd.DataFrame({"person_id": ["101", "102"], "PGS": [0.2, 0.8]})
        with patch.object(aou, "load_cached_score", return_value=first.copy()) as load:
            pilot = aou.endpoint_scores("scores.tar", disease)
        load.assert_called_once_with("scores.tar", "PGS004525")
        self.assertEqual(pilot.person_id.tolist(), ["101", "102"])
        self.assertNotIn("PGS004603", pilot)

    def test_status_uses_the_validated_default_metadata_identity(self):
        with patch("aou_status.task_account") as account, \
             patch("aou_status.urlopen", side_effect=[
                 io.BytesIO(b'{"access_token":"test-token"}'), io.BytesIO(b'{}')]) as request:
            publish_status("gs://workspace/checkpoint.tar.gz", "reading_ancestry")
        account.assert_called_once()
        token_request, upload_request = [call.args[0] for call in request.call_args_list]
        self.assertEqual(token_request.full_url,
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token")
        self.assertEqual(upload_request.headers["Authorization"], "Bearer test-token")
        self.assertEqual(upload_request.data, b"reading_ancestry\n")

    def test_status_labels_never_contain_exception_text_or_runtime_data(self):
        self.assertEqual(failure_label(ValueError("participant 123456789 bad input")), "failed_other")
        self.assertEqual(failure_label(ValueError("relatedness prune contains invalid research IDs")), "failed_prune_schema")
        with patch("aou_status.task_account") as account:
            with self.assertRaisesRegex(ValueError, "fixed public label"):
                publish_status("gs://workspace/checkpoint", "participant_123456789")
        account.assert_not_called()

    def test_published_prune_sample_id_schema_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            ancestry, prune = Path(tmp) / "ancestry.tsv", Path(tmp) / "prune.tsv"
            ancestry.write_text("research_id\tpca_features\tancestry_pred\n"
                                "101\t[0.1, 0.2]\teur\n102\t[0.3, 0.4]\tafr\n"
                                "103\t[0.5, 0.6]\tamr\n")
            prune.write_text("sample_id\n101\n103\n")
            projections = Path(tmp) / "projection_pcs.parquet"
            pd.DataFrame({"IID": ["101", "102", "103"], "PC1": [1., 3., 5.],
                          "PC2": [2., 4., 6.]}).to_parquet(projections)
            remaining = aou.read_ancestry(ancestry, prune, 2, projections)
            self.assertEqual(remaining.person_id.tolist(), ["102"])
            # The published ancestry labels supply no PC values to the model.
            np.testing.assert_allclose(remaining[["PC1", "PC2"]], [[3., 4.]])
            for content in ("research_id\n101\n", "101\n103\n", "sample_id\n101\n\n103\n",
                            "sample_id\ninvalid\n", "sample_id\n", "sample_id\textra\n101\t103\n"):
                prune.write_text(content)
                with self.assertRaisesRegex(ValueError, "relatedness prune"):
                    aou.read_ancestry(ancestry, prune, 2, projections)

    def test_smoke_fit_never_passes_outer_test_to_model_or_selects_a_score(self):
        frame = pd.DataFrame({"person_id": np.arange(200), "split_group": np.arange(200),
                              "is_train": np.arange(200) < 160,
                              "PGS004536": np.arange(200) * .1, "PGS001783": np.arange(200) * -.2})
        disease = {"candidates": ["PGS004536"]}
        with patch.object(aou, "analyze_partition", return_value={"smoke": "checked"}) as fit:
            result = aou.analyze_development(frame, disease, {"seed": 13},
                SimpleNamespace(smoke_only=True), Path("results/endpoint"), object())
        fit.assert_called_once()
        fitted = fit.call_args.args[0]
        self.assertTrue(set(fitted.person_id).isdisjoint(set(frame.loc[~frame.is_train, "person_id"])))
        np.testing.assert_array_equal(fitted.PGS, fitted.PGS004536)
        self.assertEqual(set(result), {"PGS004536"})

    def test_score_panel_excludes_upstream_aou_training(self):
        path = Path(__file__).with_name("aou_pgs_panel.json")
        panel = aou.load_score_panel(path, exploratory=True)
        self.assertEqual(set(panel["endpoints"]), {"copd", "hypertension", "obesity"})
        self.assertIn("PGS004787", panel["excluded"])
        self.assertTrue(all(len(e["candidates"]) == 1 for e in panel["endpoints"].values()))
        with self.assertRaisesRegex(ValueError, "final analysis requires completed"):
            aou.load_score_panel(path, exploratory=False)

    def test_development_never_contains_outer_test_and_is_outcome_blind(self):
        frame = pd.DataFrame({"person_id": np.arange(200), "split_group": np.arange(200) // 2,
                              "is_train": np.arange(200) < 160, "event_code": np.arange(200) % 3})
        config = {"seed": 13}
        dev = aou.development_partition(frame, config)
        self.assertTrue(set(dev.person_id).isdisjoint(set(frame.loc[~frame.is_train, "person_id"])))
        altered = frame.copy()
        altered["event_code"] = 99
        np.testing.assert_array_equal(aou.development_partition(altered, config).is_train, dev.is_train)
        self.assertEqual(dev.groupby("split_group").is_train.nunique().max(), 1)

    def test_checkpoint_integrity_native_engine_and_unsafe_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cp = StudyCheckpoint(root / "results", "gs://workspace/checkpoint", "project", "analysis@example.org", {"input": "a"}, engine_hash="build-a")
            stage = cp.root / "stage"
            stage.mkdir()
            (stage / "fit").write_text("fitted model")
            with patch.object(cp, "publish"):
                cp.complete_step(stage, ["fit"], model=True)
            self.assertTrue(cp.step_is_complete(stage, model=True))
            cp.engine_hash = "build-b"
            with self.assertRaisesRegex(ValueError, "native engine differs"):
                cp.step_is_complete(stage, model=True)
            cp.engine_hash = "build-a"
            (stage / "fit").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "missing or corrupt"):
                cp.step_is_complete(stage, model=True)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("../escape")
                tar.addfile(info)
            with self.assertRaisesRegex(ValueError, "relative regular"):
                cp.restore(archive, {"input": "a"})

    def test_loss_uncertainty_and_training_defined_pc_support(self):
        delta = np.array([1., 2., 3., 4.])
        result = loss_summary(delta, np.arange(4))
        self.assertAlmostEqual(result["brier"], 2.5)
        self.assertAlmostEqual(result["brier_standard_error"], delta.std(ddof=1) / 2)
        train = pd.DataFrame({"PC1": np.linspace(-1, 1, 40), "ancestry": "a", "sex": 0, "age0": 50})
        test = train.iloc[:3].copy()
        test.loc[test.index[-1], "PC1"] = 1000
        groups = dict(audit_groups(train, test))
        self.assertTrue(groups["pc_outside_training_support"][-1])

    def test_submission_requires_locally_configured_account_before_cli_calls(self):
        env = {"AOU_WORKSPACE_ID": "workspace", "GOOGLE_PROJECT": "project",
               "AOU_BUCKET_ID": "bucket", "WORKSPACE_BUCKET": "gs://bucket"}
        with patch.dict(submit_aou.os.environ, env, clear=True), \
             patch.object(submit_aou.Workbench, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "AOU_EXPECTED_ACCOUNT is unset"):
                submit_aou.Workbench()
        command.assert_not_called()

    def test_terminated_controller_reaps_its_fit_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_file = root / "child.pid"
            child_code = (
                "import os,time; from pathlib import Path; "
                f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
            )
            controller_code = (
                "from aou_survival import bounded_fit; "
                f"bounded_fit({[sys.executable, '-c', child_code]!r}, 20, {str(root / 'fit.log')!r})"
            )
            parent = subprocess.Popen([sys.executable, "-c", controller_code],
                                      cwd=Path(__file__).parent, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while not pid_file.exists() and time.monotonic() < deadline:
                    if parent.poll() is not None:
                        self.fail("controller exited before starting its fit")
                    time.sleep(.05)
                self.assertTrue(pid_file.exists(), "fit did not start within its startup budget")
                child_pid = int(pid_file.read_text())
                parent.send_signal(signal.SIGTERM)
                self.assertNotEqual(parent.wait(timeout=7), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait()
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_submission_stops_before_mutation_for_forbidden_account(self):
        env = {"AOU_WORKSPACE_ID": "workspace", "GOOGLE_PROJECT": "project",
               "AOU_BUCKET_ID": "bucket", "WORKSPACE_BUCKET": "gs://bucket",
               "AOU_EXPECTED_ACCOUNT": "analyst@example.org",
               "AOU_GCLOUD_CONFIGURATION": "test-profile",
               "WORKBENCH_CONTEXT_PARENT_DIR": "/configured/test-context"}
        status = json.dumps({"user": {"email": "user@example.org"}})
        with patch.dict(submit_aou.os.environ, env), \
             patch.object(submit_aou.Workbench, "command", side_effect=["analyst@example.org", status]) as command:
            with self.assertRaises(RuntimeError):
                submit_aou.Workbench()
        self.assertEqual(command.call_count, 2)
        self.assertEqual(command.call_args.args[0], ["wb", "status", "--format=JSON"])

    def test_identity_blocks_user_and_all_unapproved_human_accounts(self):
        for email in [None, "", "user@example.org", "USER@example.org",
                      "service-user@project.iam.gserviceaccount.com", "not-an-email"]:
            with self.assertRaises(RuntimeError):
                check_account(email)
        with self.assertRaises(RuntimeError):
            check_account("someone@example.org", expected="analyst@example.org")
        self.assertEqual(check_account("analyst@example.org", expected="analyst@example.org"),
                         "analyst@example.org")

    def test_competing_incidence_matches_two_exponential_causes(self):
        times = np.linspace(0, 5, 101)
        # Nonzero origin hazards cancel when conditioning on event-free entry.
        h = np.array([[3 + .2 * times], [7 + .1 * times]])
        cif = aou.cif_from_hazards(h)
        np.testing.assert_allclose(cif[0, 0], 2 / 3 * -np.expm1(-.3 * times), atol=1e-14)
        np.testing.assert_allclose(cif[1, 0], 1 / 3 * -np.expm1(-.3 * times), atol=1e-14)
        np.testing.assert_array_equal(cif[:, :, 0], 0)
        self.assertLess(cif[0, 0, -1], -np.expm1(-.2 * times[-1]))

    def test_hazards_must_be_finite_and_monotone(self):
        for h in [np.array([[[0, 1, .5]]]), np.array([[[0, np.nan]]]), np.array([[[-1, 0]]])]:
            with self.assertRaises(ValueError):
                aou.cif_from_hazards(h)

    def test_prevalence_rank_happens_before_pgs_intersection(self):
        resolved = pd.DataFrame({"concept_code": ["38341003", "44054006"],
                                 "concept_id": [10, 30], "concept_name": ["Hypertension", "T2D"]})
        ranked = pd.DataFrame({"concept_id": [20, 10, 30], "case_count": [300, 200, 100]})
        with patch.object(selection, "resolve_snomed_codes", return_value=resolved), \
             patch.object(selection, "extract_ohdsi_canonical_disease_concepts", return_value={10, 20, 30}), \
             patch.object(selection, "rank_disease_concepts_by_prevalence", return_value=ranked), \
             patch.object(selection, "_concept_names", return_value={10: "Hypertension", 20: "Unmapped"}):
            selected = selection.select_runtime_diseases(None, "project.dataset", 2, Path("unused"))
        self.assertEqual(list(selected), ["hypertension"])
        self.assertEqual(selected["hypertension"]["pgs"], "PGS001320")

    def test_incident_cohort_bounds_and_competing_death(self):
        base = pd.DataFrame({
            "person_id": list("abcdefg"), "sex_at_birth_concept_id": [45880669] * 7,
            "split_group": list("abcdefg"), "obs_start": ["2018-01-01"] * 7,
            "birth_date": ["1970-01-01"] * 7, "baseline": ["2020-01-01"] * 7,
            "obs_end": ["2022-01-01"] * 7,
            "death_date": [None, None, "2020-06-01", None, "2021-01-01", "2019-01-01", None],
        })
        scores = pd.DataFrame({"person_id": list("abcdefg"), "PGS": range(7)})
        cases = pd.DataFrame({"person_id": list("abcdefg"), "disease_date": [
            "2019-01-01", "2021-01-01", "2021-01-01", "2023-01-01", "2021-01-01", None, "2020-01-01"]})
        c = {"seed": 8, "max_rows_per_disease": 100, "train_fraction": .9,
             "lookback_days": 365, "landmark_days": 0}
        cohort = aou.build_cohort(base, scores, cases, c).set_index("person_id")
        self.assertEqual(set(cohort.index), {"b", "c", "d", "e"})
        self.assertEqual(cohort.loc["b", "event_code"], 1)
        self.assertEqual(cohort.loc["c", "event_code"], 2)
        self.assertEqual(cohort.loc["d", "event_code"], 0)
        self.assertEqual(cohort.loc["e", "event_code"], 1)
        self.assertTrue(cohort.loc["e", "disease_death_same_day"])
        self.assertAlmostEqual(cohort.loc["d", "followup"], 731 / 365.25)
        permuted = aou.build_cohort(base.iloc[::-1], scores, cases, c).set_index("person_id")
        pd.testing.assert_series_equal(cohort.is_train.sort_index(), permuted.is_train.sort_index())

    def test_final_analysis_audits_only_the_analysed_endpoint(self):
        audited = {"discovery": {"status": "no_documented_aou_development", "sources": ["https://x"]},
                   "components": {"status": "no_documented_aou_development", "sources": ["https://x"]},
                   "tuning": {"status": "no_documented_aou_development", "sources": ["https://x"]}}
        panel = {"endpoints": {"hypertension": {"candidates": ["PGS000001"]},
                               "copd": {"candidates": ["PGS000002"]}},
                 "excluded": [], "scores": {"PGS000001": {"development_audit": audited},
                                            "PGS000002": {"development_audit": None}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "panel.json"); path.write_text(json.dumps(panel))
            aou.load_score_panel(path, exploratory=False, endpoints=["hypertension"])
            aou.load_score_panel(path, exploratory=True)
            with self.assertRaisesRegex(ValueError, "provenance for PGS000002"):
                aou.load_score_panel(path, exploratory=False)
            with self.assertRaisesRegex(ValueError, "not in the prespecified panel"):
                aou.load_score_panel(path, exploratory=False, endpoints=["stroke"])

    def test_landmark_excludes_enrollment_diagnoses_and_restarts_follow_up(self):
        base = pd.DataFrame({
            "person_id": list("abcd"), "sex_at_birth_concept_id": [45880669] * 4,
            "split_group": list("abcd"), "obs_start": ["2018-01-01"] * 4,
            "birth_date": ["1970-01-01"] * 4, "baseline": ["2020-01-01"] * 4,
            "obs_end": ["2023-01-01", "2023-01-01", "2020-03-01", "2023-01-01"],
            "death_date": [None, None, None, "2020-04-01"],
        })
        scores = pd.DataFrame({"person_id": list("abcd"), "PGS": range(4)})
        cases = pd.DataFrame({"person_id": list("abcd"),
                              "disease_date": ["2020-03-01", "2021-02-04", None, None]})
        c = {"seed": 8, "max_rows_per_disease": 100, "train_fraction": .9,
             "lookback_days": 365, "landmark_days": 180}
        cohort = aou.build_cohort(base, scores, cases, c).set_index("person_id")
        # a: diagnosed inside the landmark window (prevalent at detection); c: lost; d: died.
        self.assertEqual(set(cohort.index), {"b"})
        self.assertEqual(cohort.loc["b", "event_code"], 1)
        self.assertAlmostEqual(cohort.loc["b", "followup"], (400 - 180) / 365.25)
        self.assertAlmostEqual(cohort.loc["b", "age0"], (pd.Timestamp("2020-06-29") - pd.Timestamp("1970-01-01")).days / 365.25)

    def test_weighted_auc_ranks_cases_over_controls_with_ties_at_half(self):
        self.assertEqual(aou.weighted_auc([.1, .2, .9, .8], [0, 0, 1, 1], [1, 1, 1, 1]), 1.0)
        self.assertEqual(aou.weighted_auc([.5, .5, .5, .5], [0, 1, 0, 1], [1, 1, 1, 1]), 0.5)
        self.assertAlmostEqual(aou.weighted_auc([.1, .9, .5], [0, 1, 1], [1, 1, 2]), (1 + 2) / 3)
        self.assertAlmostEqual(aou.weighted_auc([.1, .9, .5], [0, 1, 1], [1, 3, 1]), (3 + 1) / 4)
        with self.assertRaises(ValueError):
            aou.weighted_auc([.1, .9], [1, 1], [1, 1])

    def test_incremental_value_is_the_paired_brier_difference(self):
        rng = np.random.default_rng(3)
        n = 400
        test = pd.DataFrame({"event_code": rng.integers(0, 2, n), "followup": rng.uniform(.1, 4, n),
                             "ancestry": ["eur"] * n, "sex": rng.integers(0, 2, n),
                             "age0": rng.uniform(20, 70, n), "split_group": [f"g{i % 8}" for i in range(n)],
                             "PC1": rng.normal(size=n), "PC2": rng.normal(size=n)})
        train = test.copy()
        full = np.clip(rng.uniform(.01, .5, (n, 1)), 0, 1)
        null = np.full((n, 1), float(full.mean()))
        with patch.object(aou, "ipcw_weights", return_value=np.ones(n)):
            rows = aou.incremental_value(train, test, full, null, [1.0], 20)
            reference = {(r["group"], r["horizon"]): r for r in aou.evaluate(train, test, full, [1.0], 20)}
            baseline = {(r["group"], r["horizon"]): r for r in aou.evaluate(train, test, null, [1.0], 20)}
        overall = next(r for r in rows if r["group"] == "overall")
        self.assertAlmostEqual(overall["brier_difference"],
                               reference[("overall", 1.0)]["brier"] - baseline[("overall", 1.0)]["brier"])
        self.assertAlmostEqual(overall["auc_difference"],
                               reference[("overall", 1.0)]["ipcw_auc"] - baseline[("overall", 1.0)]["ipcw_auc"])
        self.assertGreater(overall["brier_difference_standard_error"], 0)

    def test_score_api_does_not_use_ctn_mean_prediction(self):
        from unittest.mock import Mock
        data = pd.DataFrame({"PGS": [2., 5.], "PC1": [-1., 1.],
                             "baseline": pd.to_datetime(["2020-01-01", "2021-01-01"]),
                             "death_date": [pd.NaT, pd.NaT], "event_code": [1, 0]})
        model = Mock()
        model.transformation_score.return_value = np.array([-.5, .5])
        np.testing.assert_array_equal(transforms.transformed_score(model, "ctn", data, 1), [-.5, .5])
        pd.testing.assert_frame_equal(model.transformation_score.call_args.args[0], data[["PGS", "PC1"]])
        model.predict.assert_not_called()
        with self.assertRaisesRegex(ValueError, "frozen external CTN"):
            transforms.transformed_score(model, "location_scale", data, 1)

    def test_enrollment_lookback_is_required_without_future_survival_requirement(self):
        count = 20
        base = pd.DataFrame({"person_id": [str(i) for i in range(count)],
                             "split_group": [str(i) for i in range(count)],
                             "sex_at_birth_concept_id": [8507] * count,
                             "birth_date": ["1970-01-01"] * count,
                             "baseline": ["2020-01-01"] * count,
                             "obs_start": ["2018-01-01"] * count,
                             "obs_end": ["2020-01-02"] * count, "death_date": [None] * count})
        base.loc[0, "obs_start"] = "2019-12-01"
        scores = pd.DataFrame({"person_id": base.person_id, "PGS": np.arange(count)})
        cases = pd.DataFrame({"person_id": base.person_id, "disease_date": [None] * count})
        config = {"lookback_days": 365, "landmark_days": 0, "seed": 1, "train_fraction": .8,
                  "max_rows_per_disease": 100}
        cohort = aou.build_cohort(base, scores, cases, config)
        self.assertEqual(len(cohort), count - 1)
        self.assertNotIn("0", set(cohort.person_id))
        np.testing.assert_allclose(cohort.followup, 1 / 365.25)

    def test_reverse_km_ties_and_early_censoring(self):
        train = pd.DataFrame({"followup": [1, 1, 2, 3] * 10,
                              "event_code": [1, 0, 2, 1] * 10, "ancestry": ["A"] * 40})
        times, g = aou.censor_km(train)
        np.testing.assert_allclose(times, [1, 2, 3])
        np.testing.assert_allclose(g, [2 / 3, 2 / 3, 2 / 3])
        test = pd.DataFrame({"followup": [.5, 1, 1.5, 2.5], "event_code": [0, 1, 2, 0],
                             "ancestry": ["A"] * 4})
        np.testing.assert_allclose(aou.ipcw_weights(train, test, 2), [0, 1, 1.5, 1.5])
        with self.assertRaises(ValueError):
            aou.ipcw_weights(train, test, 4)

    def test_cached_score_uses_identified_column_and_rejects_duplicate_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            score = root / "arrays_PGS001320.sscore"
            score.write_text("#IID\tPGS001320_AVG\tPGS001320_MISSING_PCT\n1\t0.25\t0\n2\t-0.5\t2\n")
            archive = root / "scores.tar"
            with tarfile.open(archive, "w") as tar:
                tar.add(score, arcname=score.name)
            shared = root / "shared_features.tar.gz"
            projection = root / "projection_pcs.parquet"
            projection.write_bytes(b"projection artifact")
            with tarfile.open(shared, "w:gz") as tar:
                tar.add(archive, arcname="shared_features/scores.tar")
                tar.add(projection, arcname="shared_features/projection_pcs.parquet")
            projected = root / "extracted.parquet"
            extracted = aou.unpack_score_cache(shared, root / "extracted.tar", projected)
            self.assertEqual(extracted.read_bytes(), archive.read_bytes())
            self.assertEqual(projected.read_bytes(), projection.read_bytes())
            for members in ([], ["first/scores.tar", "second/scores.tar"]):
                with tarfile.open(shared, "w:gz") as tar:
                    for member in members:
                        tar.add(archive, arcname=member)
                with self.assertRaisesRegex(ValueError, "exactly one scores.tar"):
                    aou.unpack_score_cache(shared, extracted, projected)
                self.assertEqual(extracted.read_bytes(), archive.read_bytes())
                self.assertEqual(list(root.glob("scores-*.partial")), [])
            actual = aou.load_cached_score(archive, "PGS001320")
            self.assertEqual(actual.person_id.tolist(), ["1", "2"])
            self.assertEqual(actual.PGS.tolist(), [.25, -.5])
            self.assertEqual(actual.PGS001320_MISSING_PCT.tolist(), [0, 2])
            with tarfile.open(archive, "a") as tar:
                tar.add(score, arcname="second.sscore")
            with self.assertRaises(ValueError):
                aou.load_cached_score(archive, "PGS001320")

    def test_cached_score_rejects_missingness_before_transformation(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "scores.tar"
            for content in ("#IID\tPGS001320_AVG\n1\t0\n", *[
                    f"#IID\tPGS001320_AVG\tPGS001320_MISSING_PCT\n1\t0\t{value}\n"
                    for value in (100, -1, 101, "nan")]):
                with tarfile.open(archive, "w") as tar:
                    payload = content.encode()
                    item = tarfile.TarInfo("score.sscore")
                    item.size = len(payload)
                    tar.addfile(item, io.BytesIO(payload))
                with self.assertRaises(ValueError):
                    aou.load_cached_score(archive, "PGS001320")

    def test_analysis_has_no_fake_deployment_values(self):
        config = json.loads(Path(__file__).with_name("aou_analysis.json").read_text())
        self.assertNotIn("workspace_cdr", config)
        self.assertNotIn("google_project", config)
        config.update(google_project="wb-project", workspace_cdr="cdr-project.release")
        aou.validate_config(config)
        config["horizons_years"] = [1, float("nan")]
        with self.assertRaises(ValueError):
            aou.validate_config(config)


if __name__ == "__main__":
    unittest.main()
