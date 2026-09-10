"""Small deterministic tests; run on MSI, never against participant data."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import aou_survival as aou
import submit_aou
from aou_identity import check_account
import disease_selection as selection


class SurvivalContractTests(unittest.TestCase):
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
            "birth_date": ["1970-01-01"] * 7, "baseline": ["2020-01-01"] * 7,
            "obs_end": ["2022-01-01"] * 7,
            "death_date": [None, None, "2020-06-01", None, "2021-01-01", "2019-01-01", None],
        })
        scores = pd.DataFrame({"person_id": list("abcdefg"), "PGS": range(7)})
        cases = pd.DataFrame({"person_id": list("abcdefg"), "disease_date": [
            "2019-01-01", "2021-01-01", "2021-01-01", "2023-01-01", "2021-01-01", None, "2020-01-01"]})
        c = {"seed": 8, "max_rows_per_disease": 100, "train_fraction": .8}
        cohort = aou.build_cohort(base, scores, cases, c).set_index("person_id")
        self.assertEqual(set(cohort.index), {"b", "c", "d"})
        self.assertEqual(cohort.loc["b", "event_code"], 1)
        self.assertEqual(cohort.loc["c", "event_code"], 2)
        self.assertEqual(cohort.loc["d", "event_code"], 0)
        self.assertAlmostEqual(cohort.loc["d", "followup"], 731 / 365.25)
        permuted = aou.build_cohort(base.iloc[::-1], scores, cases, c).set_index("person_id")
        pd.testing.assert_series_equal(cohort.is_train.sort_index(), permuted.is_train.sort_index())

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
