"""
Fast pruning-and-thresholding (P+T) PRS wrapper built on PLINK2.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class PPlusT:
    def __init__(
        self,
        threads: int | None = None,
        plink_memory_mb: int | None = None,
        p_threshold: float = 0.05,
        p_thresholds: list[float] | None = None,
        prune_window_kb: int = 200,
        prune_step: int = 50,
        prune_r2: float = 0.2,
        fallback_top_k: int = 5000,
    ):
        env_threads = os.environ.get("GCTB_THREADS")
        if threads is None and env_threads:
            try:
                threads = int(env_threads)
            except Exception:
                threads = None
        self.threads = int(threads) if threads is not None else 4
        env_mem = os.environ.get("PLINK_MEMORY_MB")
        if plink_memory_mb is None and env_mem:
            try:
                plink_memory_mb = int(env_mem)
            except Exception:
                plink_memory_mb = None
        self.plink_memory_mb = int(plink_memory_mb) if plink_memory_mb is not None else None
        self.p_threshold = float(p_threshold)
        self.p_thresholds = list(p_thresholds) if p_thresholds is not None else [1e-4, 1e-3, 1e-2, 1e-1, 5e-1]
        self.prune_window_kb = int(prune_window_kb)
        self.prune_step = int(prune_step)
        self.prune_r2 = float(prune_r2)
        self.fallback_top_k = int(fallback_top_k)

    def _plink(self) -> str:
        return shutil.which("plink2") or "plink2"

    def _run(self, cmd: list[str], desc: str) -> None:
        print(f"[P+T] {desc}: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"P+T command failed ({desc}) rc={res.returncode}\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
            )

    def _common(self) -> list[str]:
        out = ["--threads", str(self.threads)]
        if self.plink_memory_mb is not None and self.plink_memory_mb > 0:
            out.extend(["--memory", str(self.plink_memory_mb)])
        return out

    @staticmethod
    def _read_sscore(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=r"\s+")
        iid_col = "IID" if "IID" in df.columns else ("#IID" if "#IID" in df.columns else None)
        if iid_col is None:
            raise RuntimeError(f"P+T .sscore missing IID column: {path}; cols={list(df.columns)}")
        score_col = None
        for c in ("SCORE1_AVG", "SCORE1_SUM", "SCORE1", "SCORE"):
            if c in df.columns:
                score_col = c
                break
        if score_col is None:
            score_like = [c for c in df.columns if "SCORE" in c.upper()]
            if not score_like:
                raise RuntimeError(f"P+T .sscore missing SCORE column: {path}; cols={list(df.columns)}")
            score_col = score_like[0]
        out = df[[iid_col, score_col]].copy()
        out.columns = ["IID", "PRS"]
        out["IID"] = out["IID"].astype(str)
        out["PRS"] = pd.to_numeric(out["PRS"], errors="coerce").astype(float)
        return out

    @staticmethod
    def _find_qscore_output(out_prefix: str, label: str) -> str:
        cands = [
            f"{out_prefix}.{label}.sscore",
            f"{out_prefix}.{label}.profile",
            f"{out_prefix}.{label}.sscore.zst",
        ]
        for p in cands:
            if Path(p).exists():
                return p
        raise RuntimeError(
            f"P+T q-score-range output not found for label={label}; looked for {cands}"
        )

    @staticmethod
    def _read_binary_pheno(path: str) -> pd.DataFrame:
        ph = pd.read_csv(path, sep=r"\s+", header=None, names=["FID", "IID", "y"])
        ph["IID"] = ph["IID"].astype(str)
        ph["y"] = pd.to_numeric(ph["y"], errors="coerce").astype(float)
        return ph[["IID", "y"]].dropna()

    @staticmethod
    def _train_metrics(train_scores: pd.DataFrame, pheno: pd.DataFrame) -> tuple[float, float, int]:
        m = pheno.merge(train_scores, on="IID", how="inner")
        m = m.dropna(subset=["y", "PRS"])
        n = int(len(m))
        if n == 0:
            return float("nan"), float("nan"), 0
        y = m["y"].to_numpy(dtype=int)
        x = m["PRS"].to_numpy(dtype=float).reshape(-1, 1)
        uniq = np.unique(y)
        if uniq.size < 2:
            return 1.0, float("nan"), n
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(x, y)
        p = lr.predict_proba(x)[:, 1]
        yhat = (p >= 0.5).astype(int)
        acc = float(accuracy_score(y, yhat))
        auc = float(roc_auc_score(y, p))
        return acc, auc, n

    def fit_and_predict(
        self,
        bfile_train: str,
        bfile_test: str,
        pheno_file: str,
        covar_file: str | None,
        freq_file: str | None,
        out_prefix: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        plink = self._plink()
        common = self._common()
        if covar_file is None or not Path(covar_file).exists():
            raise FileNotFoundError(
                "P+T requires a covariate file with PCs for GWAS effect-size estimation. "
                f"Missing covar_file={covar_file}"
            )

        prune_prefix = f"{out_prefix}.prune"
        keep_ids: set[str] | None = None
        try:
            self._run(
                [
                    plink,
                    "--bfile",
                    bfile_train,
                    "--indep-pairwise",
                    str(self.prune_window_kb),
                    str(self.prune_step),
                    str(self.prune_r2),
                    *common,
                    "--out",
                    prune_prefix,
                ],
                "LD pruning",
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "less than 50 samples" in msg:
                print("[P+T] LD pruning skipped (train n<50); using all variants for thresholding")
                bim_path = Path(f"{bfile_train}.bim")
                if not bim_path.exists():
                    raise
                keep_ids = set()
                with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) >= 2 and parts[1]:
                            keep_ids.add(parts[1])
            else:
                raise

        glm_prefix = f"{out_prefix}.gwas"
        glm_cmd = [
            plink,
            "--bfile",
            bfile_train,
            "--pheno",
            pheno_file,
            "--glm",
            "hide-covar",
            "--allow-no-sex",
            *common,
            "--out",
            glm_prefix,
        ]
        glm_cmd.extend(["--covar", str(covar_file)])
        fallback_weights: pd.DataFrame | None = None
        try:
            self._run(glm_cmd, "GWAS")
            glm_files = sorted(Path(".").glob(f"{Path(glm_prefix).name}*.glm.*"))
            if not glm_files:
                glm_files = sorted(Path(Path(glm_prefix).parent).glob(f"{Path(glm_prefix).name}*.glm.*"))
            if not glm_files:
                raise RuntimeError(f"P+T GWAS output not found for prefix={glm_prefix}")
            glm_path = str(glm_files[0])
            print(f"[P+T] using GWAS results: {glm_path}")

            gwas = pd.read_csv(glm_path, sep=r"\s+")
            if "TEST" in gwas.columns:
                gwas = gwas[gwas["TEST"].astype(str) == "ADD"].copy()
            if "P" not in gwas.columns or "ID" not in gwas.columns or "A1" not in gwas.columns:
                raise RuntimeError(f"P+T GWAS file missing required columns (ID/A1/P): {glm_path}")
            if "BETA" in gwas.columns:
                eff = pd.to_numeric(gwas["BETA"], errors="coerce")
            elif "OR" in gwas.columns:
                orv = pd.to_numeric(gwas["OR"], errors="coerce")
                eff = np.log(orv.where(orv > 0))
            else:
                raise RuntimeError(f"P+T GWAS file missing BETA/OR effect column: {glm_path}")

            gwas = pd.DataFrame(
                {
                    "ID": gwas["ID"].astype(str),
                    "A1": gwas["A1"].astype(str),
                    "P": pd.to_numeric(gwas["P"], errors="coerce"),
                    "EFF": eff,
                }
            ).dropna(subset=["P", "EFF"])
        except RuntimeError as e:
            msg = str(e).lower()
            if "all samples for --glm phenotype" not in msg:
                raise
            print("[P+T] GWAS skipped (phenotype has one class in training); using deterministic fallback weights")
            bim = pd.read_csv(
                f"{bfile_train}.bim",
                sep=r"\s+",
                header=None,
                names=["CHR", "ID", "CM", "POS", "A1", "A2"],
                dtype={"ID": str, "A1": str},
            )
            bim = bim[bim["ID"].isin(keep_ids)].copy()
            if bim.empty:
                raise RuntimeError("P+T fallback failed: no variants available in training .bim")
            bim = bim.head(max(1, self.fallback_top_k)).copy()
            signs = np.where((np.arange(len(bim)) % 2) == 0, 1.0, -1.0)
            fallback_weights = pd.DataFrame(
                {
                    "ID": bim["ID"].astype(str),
                    "A1": bim["A1"].astype(str),
                    "EFF": signs * 1e-6,
                }
            )

        if keep_ids is None:
            prune_in = Path(f"{prune_prefix}.prune.in")
            if not prune_in.exists():
                raise RuntimeError(f"P+T missing prune list: {prune_in}")
            keep_ids = set(x.strip() for x in prune_in.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip())
        if fallback_weights is None:
            gwas = gwas[gwas["ID"].isin(keep_ids)].copy()
            gwas = gwas.sort_values("P")
        else:
            gwas = None

        pheno = self._read_binary_pheno(pheno_file)
        thresholds = sorted(set(float(x) for x in self.p_thresholds))
        records: list[dict[str, object]] = []
        train_scores_by_thr: dict[float, pd.DataFrame] = {}

        if fallback_weights is None:
            all_w = gwas[["ID", "A1", "EFF", "P"]].copy()
        else:
            all_w = fallback_weights.copy()
            all_w["P"] = 0.0

        # Single score file + single p-value file, then compute all thresholds with q-score-range in one PLINK pass.
        score_file_all = f"{out_prefix}.all.score"
        qfile = f"{out_prefix}.qvals.tsv"
        rfile = f"{out_prefix}.ranges.tsv"
        all_w[["ID", "A1", "EFF"]].to_csv(score_file_all, sep="\t", header=True, index=False)
        all_w[["ID", "P"]].to_csv(qfile, sep="\t", header=True, index=False)
        labels: dict[float, str] = {}
        with open(rfile, "w", encoding="utf-8") as f:
            f.write("RANGE\tLOW\tHIGH\n")
            for thr in thresholds:
                label = f"T{str(thr).replace('.', 'p')}"
                labels[thr] = label
                f.write(f"{label}\t0\t{thr}\n")

        train_out = f"{out_prefix}.qtrain"
        train_cmd = [
            plink,
            "--bfile",
            bfile_train,
            "--score",
            score_file_all,
            "1",
            "2",
            "3",
            "header",
            "--q-score-range",
            rfile,
            qfile,
            "1",
            "2",
            "header",
            "--allow-no-sex",
            *common,
            "--out",
            train_out,
        ]
        if freq_file is not None and Path(freq_file).exists():
            train_cmd.extend(["--read-freq", freq_file])
        self._run(train_cmd, "score train all thresholds (q-score-range)")

        for thr in thresholds:
            label = labels[thr]
            out_path = self._find_qscore_output(train_out, label)
            train_scores = self._read_sscore(out_path)
            train_scores_by_thr[thr] = train_scores
            n_snps = int(np.count_nonzero(all_w["P"].to_numpy(dtype=float) <= thr))
            if n_snps == 0:
                n_snps = 1
            train_acc, train_auc, n_train_used = self._train_metrics(train_scores, pheno)
            records.append(
                {
                    "p_threshold": thr,
                    "n_snps": n_snps,
                    "n_train_used": int(n_train_used),
                    "train_accuracy": train_acc,
                    "train_auc": train_auc,
                }
            )

        metrics = pd.DataFrame(records).sort_values("p_threshold").reset_index(drop=True)
        metrics_rank = metrics.copy()
        metrics_rank["train_accuracy"] = metrics_rank["train_accuracy"].fillna(-np.inf)
        metrics_rank["train_auc"] = metrics_rank["train_auc"].fillna(-np.inf)
        best_idx = metrics_rank.sort_values(["train_accuracy", "train_auc", "n_snps"], ascending=[False, False, False]).index[0]
        best_thr = float(metrics.loc[best_idx, "p_threshold"])
        print(
            f"[P+T] selected best threshold={best_thr:g} "
            f"(train_accuracy={metrics.loc[best_idx, 'train_accuracy']}, "
            f"train_auc={metrics.loc[best_idx, 'train_auc']}, "
            f"n_snps={int(metrics.loc[best_idx, 'n_snps'])})"
        )

        best_train_scores = train_scores_by_thr[best_thr]
        if fallback_weights is None:
            best_w = all_w[all_w["P"] <= best_thr].copy()
            if best_w.empty:
                best_w = all_w.nsmallest(1, "P").copy()
        else:
            best_w = all_w.copy()
        best_score_file = f"{out_prefix}.best.score"
        best_w[["ID", "A1", "EFF"]].to_csv(best_score_file, sep="\t", header=False, index=False)
        test_out = f"{out_prefix}.best.test"
        test_cmd = [plink, "--bfile", bfile_test, "--score", best_score_file, "1", "2", "3", "--allow-no-sex", *common, "--out", test_out]
        if freq_file is not None and Path(freq_file).exists():
            test_cmd.extend(["--read-freq", freq_file])
        self._run(test_cmd, f"score test best p={best_thr:g}")
        best_test_scores = self._read_sscore(f"{test_out}.sscore")

        meta = {
            "best_p_threshold": best_thr,
            "best_train_accuracy": float(metrics.loc[best_idx, "train_accuracy"]),
            "best_train_auc": float(metrics.loc[best_idx, "train_auc"]),
            "best_n_snps": int(metrics.loc[best_idx, "n_snps"]),
            "threshold_metrics": metrics,
        }
        return best_train_scores, best_test_scores, meta
