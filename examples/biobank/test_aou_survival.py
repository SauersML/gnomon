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
from aou_checkpoint import StudyCheckpoint
from aou_evaluation import audit_groups, paired_loss_summary


class SurvivalContractTests(unittest.TestCase):
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
        reports = {pgs: {"models": {"pc_varying_ctn": {"metrics": metrics}}}
                   for pgs in ("PGS004525", "PGS004603")}
        with self.assertRaisesRegex(ValueError, "supported development Brier"):
            aou.select_development_score(reports, [1.])
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

    def test_primary_pilot_does_not_depend_on_challenger_availability(self):
        disease = {"candidates": ["PGS004525", "PGS004603"]}
        first = pd.DataFrame({"person_id": ["101", "102"], "PGS": [0.2, 0.8]})
        second = pd.DataFrame({"person_id": ["102"], "PGS": [-0.4]})
        with patch.object(aou, "load_cached_score", return_value=first.copy()) as load:
            pilot = aou.endpoint_scores("scores.tar", disease, primary_only=True)
        load.assert_called_once_with("scores.tar", "PGS004525")
        self.assertEqual(pilot.person_id.tolist(), ["101", "102"])
        self.assertNotIn("PGS004603", pilot)
        with patch.object(aou, "load_cached_score", side_effect=[first.copy(), second]):
            comparison = aou.endpoint_scores("scores.tar", disease, primary_only=False)
        self.assertEqual(comparison.person_id.tolist(), ["102"])
        self.assertEqual(comparison.PGS004525.tolist(), [0.8])
        self.assertEqual(comparison.PGS004603.tolist(), [-0.4])

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
            remaining = aou.read_ancestry(ancestry, prune, 2)
            self.assertEqual(remaining.person_id.tolist(), ["102"])
            np.testing.assert_allclose(remaining[["PC1", "PC2"]], [[.3, .4]])
            for content in ("research_id\n101\n", "101\n103\n", "sample_id\n101\n\n103\n",
                            "sample_id\ninvalid\n", "sample_id\n", "sample_id\textra\n101\t103\n"):
                prune.write_text(content)
                with self.assertRaisesRegex(ValueError, "relatedness prune"):
                    aou.read_ancestry(ancestry, prune, 2)

    def test_smoke_fit_never_passes_outer_test_to_model_or_selects_a_score(self):
        frame = pd.DataFrame({"person_id": np.arange(200), "split_group": np.arange(200),
                              "is_train": np.arange(200) < 160,
                              "PGS004536": np.arange(200) * .1, "PGS001783": np.arange(200) * -.2})
        disease = {"candidates": ["PGS004536", "PGS001783"]}
        with patch.object(aou, "analyze_partition", return_value=({"smoke": "checked"}, {})) as fit, \
             patch.object(aou, "select_development_score") as select:
            result = aou.analyze_development(frame, disease, {"seed": 13, "crossfit_folds": 2},
                SimpleNamespace(smoke_only=True), Path("results/endpoint"), object())
        fit.assert_called_once()
        fitted = fit.call_args.args[0]
        self.assertTrue(set(fitted.person_id).isdisjoint(set(frame.loc[~frame.is_train, "person_id"])))
        np.testing.assert_array_equal(fitted.PGS, fitted.PGS004536)
        self.assertEqual(set(result), {"PGS004536"})
        select.assert_not_called()

    def test_score_panel_excludes_upstream_aou_training(self):
        panel = aou.load_score_panel(Path(__file__).with_name("aou_pgs_panel.json"))
        self.assertEqual(set(panel["endpoints"]), {"copd", "hypertension", "obesity"})
        self.assertIn("PGS004787", panel["excluded"])
        self.assertTrue(all(len(e["candidates"]) == 2 for e in panel["endpoints"].values()))

    def test_development_never_contains_outer_test_and_is_outcome_blind(self):
        frame = pd.DataFrame({"person_id": np.arange(200), "split_group": np.arange(200) // 2,
                              "is_train": np.arange(200) < 160, "event_code": np.arange(200) % 3})
        config = {"seed": 13, "crossfit_folds": 2}
        dev = aou.development_partition(frame, config)
        self.assertTrue(set(dev.person_id).isdisjoint(set(frame.loc[~frame.is_train, "person_id"])))
        altered = frame.copy()
        altered["event_code"] = 99
        np.testing.assert_array_equal(aou.development_partition(altered, config).is_train, dev.is_train)
        self.assertEqual(dev.groupby("split_group").is_train.nunique().max(), 1)

    def test_selection_uses_supported_development_probability_loss(self):
        def report(loss, status="ok"):
            return {"models": {"pc_varying_ctn": {"metrics": [
                {"group": "overall", "horizon": 3., "status": status, "brier": loss}]}}}
        selected, losses = aou.select_development_score({"PGS004536": report(.1), "PGS001783": report(.09)}, [3.])
        self.assertEqual(selected, "PGS001783")
        with self.assertRaisesRegex(ValueError, "supported development"):
            aou.select_development_score({"PGS004536": report(.1), "PGS001783": report(.09, "insufficient_support")}, [3.])

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

    def test_score_assembly_rejects_overlap_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.npz"
            np.savez(path, rows=np.array([0, 1]), z=np.array([-.1, .1]))
            with self.assertRaisesRegex(ValueError, "every row exactly once"):
                transforms.assemble_scores(pd.DataFrame(index=range(3)), [path])
            with self.assertRaisesRegex(ValueError, "overlapping"):
                transforms.assemble_scores(pd.DataFrame(index=range(2)), [path, path])

    def test_paired_loss_and_training_defined_pc_support(self):
        delta = np.array([1., 2., 3., 4.])
        result = paired_loss_summary(delta, np.zeros(4), np.ones(4, bool), np.arange(4))
        self.assertAlmostEqual(result["brier_improvement"], 2.5)
        self.assertAlmostEqual(result["standard_error"], delta.std(ddof=1) / 2)
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
             "lookback_days": 365, "crossfit_folds": 2}
        cohort = aou.build_cohort(base, scores, cases, c).set_index("person_id")
        self.assertEqual(set(cohort.index), {"b", "c", "d"})
        self.assertEqual(cohort.loc["b", "event_code"], 1)
        self.assertEqual(cohort.loc["c", "event_code"], 2)
        self.assertEqual(cohort.loc["d", "event_code"], 0)
        self.assertAlmostEqual(cohort.loc["d", "followup"], 731 / 365.25)
        permuted = aou.build_cohort(base.iloc[::-1], scores, cases, c).set_index("person_id")
        pd.testing.assert_series_equal(cohort.is_train.sort_index(), permuted.is_train.sort_index())

    def test_inner_folds_keep_groups_together_and_ignore_row_order(self):
        groups = np.array(["family-a", "family-b", "family-a", "family-c", "family-d"])
        folds = transforms.grouped_folds(groups, 2, 81)
        self.assertEqual(folds[0], folds[2])
        np.testing.assert_array_equal(folds, transforms.grouped_folds(groups[::-1], 2, 81)[::-1])
        with self.assertRaises(ValueError):
            transforms.grouped_folds(["one", "one"], 2, 81)

    def test_score_api_does_not_use_ctn_mean_prediction(self):
        from unittest.mock import Mock
        data = pd.DataFrame({"PGS": [2., 5.]})
        model = Mock()
        model.transformation_score.return_value = np.array([-.5, .5])
        np.testing.assert_array_equal(transforms.transformed_score(model, "ctn", data), [-.5, .5])
        model.predict.assert_not_called()
        model.predict.return_value = {"mean_plugin": [1., 3.], "noise_scale": [2., 4.]}
        np.testing.assert_array_equal(transforms.transformed_score(model, "location_scale", data), [.5, .5])

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
        config = {"lookback_days": 365, "seed": 1, "train_fraction": .8,
                  "max_rows_per_disease": 100, "crossfit_folds": 2}
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
            score.write_text("#IID\tPGS001320_AVG\n1\t0.25\n2\t-0.5\n")
            archive = root / "scores.tar"
            with tarfile.open(archive, "w") as tar:
                tar.add(score, arcname=score.name)
            shared = root / "shared_features.tar.gz"
            with tarfile.open(shared, "w:gz") as tar:
                tar.add(archive, arcname="shared_features/scores.tar")
            extracted = aou.unpack_score_cache(shared, root / "extracted.tar")
            self.assertEqual(extracted.read_bytes(), archive.read_bytes())
            actual = aou.load_cached_score(archive, "PGS001320")
            self.assertEqual(actual.person_id.tolist(), ["1", "2"])
            self.assertEqual(actual.PGS.tolist(), [.25, -.5])
            with tarfile.open(archive, "a") as tar:
                tar.add(score, arcname="second.sscore")
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
