"""Failure and API contracts; run on MSI, without a scientific benchmark."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import analyze_results
import fit_binary
import run_study


class StudyContractTests(unittest.TestCase):
    def test_gam_uses_signed_slope_and_default_posterior_prediction(self):
        engine = Mock()
        engine.fit.return_value.predict.return_value = np.array([.2, .8])
        frame = pd.DataFrame({"y_binary": [0, 1], "PGS_z": [-1., 1.], "PC1": [0., 1.]})
        with patch.object(fit_binary, "gamfit", engine):
            actual = fit_binary.fit_gamfit(frame, frame, ["PC1"], 8)
        self.assertIn("slope_formula", engine.fit.call_args.kwargs)
        self.assertNotIn("logslope_formula", engine.fit.call_args.kwargs)
        self.assertEqual(engine.fit.return_value.predict.call_args.kwargs, {})
        np.testing.assert_array_equal(actual, [.2, .8])

    def test_failed_required_method_is_not_a_complete_replicate(self):
        frame = pd.DataFrame({"split_role": ["fit"] * 4 + ["test"] * 4,
                              "PC1": np.arange(8), "y_binary": [0, 1] * 4,
                              "PGS_z": np.arange(8), "p_true": [.2, .8] * 4})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["fit_binary", "--data", "unused", "--dem", "serial1d", "--pheno", "phenoA",
                    "--out-acc", str(root / "acc.csv"), "--out-cal", str(root / "cal.csv"),
                    "--out-status", str(root / "status.json")]
            fitters = {m: lambda *_: np.array([.2, .8, .2, .8]) for m in fit_binary._FITTERS}
            with (patch.object(sys, "argv", args),
                  patch.object(fit_binary.common, "load_normalized", return_value=frame),
                  patch.object(fit_binary.common, "ancestry_bins", return_value=[]),
                  patch.object(fit_binary, "global_discrimination",
                               return_value={"auc": 1., "brier": .04, "liability_r2": .5}),
                  patch.dict(fit_binary._FITTERS, fitters),
                  patch.object(fit_binary, "fit_gamfit", side_effect=RuntimeError("native failure")),
                  contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
                with self.assertRaisesRegex(RuntimeError, "required methods failed"):
                    fit_binary.main()
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["methods"]["gamfit"], "failed")
            self.assertEqual(status["methods"]["linpc"], "complete")

    def test_runner_failure_propagates(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(subprocess.CalledProcessError):
                run_study.run_logged([sys.executable, "-c", "raise SystemExit(3)"], Path(tmp) / "run.log")

    def test_aggregation_refuses_partial_csv_even_with_success_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "binary").mkdir()
            path = root / "binary/serial1d_phenoA_s1_acc.csv"
            pd.DataFrame({"method": ["linpc"], "metric": ["auc"], "value": [.5]}).to_csv(path, index=False)
            receipt = path.with_name("serial1d_phenoA_s1_status.json")
            receipt.write_text(json.dumps({"status": "complete", "methods": {
                m: "complete" for m in fit_binary.METHODS}}))
            with patch.object(analyze_results, "RES", root):
                with self.assertRaisesRegex(ValueError, "required method missing"):
                    analyze_results._concat("binary", "acc")


if __name__ == "__main__":
    unittest.main()
