"""study.py machinery: checkpoint resume, the fit pool, digest suppression and the
LOGO provenance gate. MSI only (python tests/test_study_machinery.py, or pytest)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from study import digest  # noqa: E402
from study.checkpoint import Checkpoint, LocalStore  # noqa: E402
from study.pool import Job, run_jobs  # noqa: E402

REGISTRY = digest.Registry({"n": {"type": "count"}, "cases": {"type": "count"}, "auc": {"type": "score"},
                            "observed_risk": {"type": "proportion", "of": "cases", "per": "n"}})


def raises(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return True
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


# ----------------------------------------------------------------- checkpoint
def seal(checkpoint, step, text):
    directory = checkpoint.begin(step)
    (directory / "value.txt").write_text(text)
    checkpoint.complete(step)


def test_checkpoint_restores_from_the_store_alone(tmp_path):
    store = LocalStore(tmp_path / "store")
    first = Checkpoint(tmp_path / "a", store, {"config": 1}, min_interval=0)
    seal(first, "fits/x/pooled", "1")
    seal(first, "fits/x/logo", "2")
    first.close()
    # A fresh machine: only the store survives.
    second = Checkpoint(tmp_path / "b", store, {"config": 1}, min_interval=0)
    assert second.restored_batches >= 1
    assert second.done("fits/x/pooled") and second.done("fits/x/logo")
    assert (second.path("fits/x/logo") / "value.txt").read_text() == "2"
    second.close()


def test_checkpoint_refuses_another_signature(tmp_path):
    store = LocalStore(tmp_path / "store")
    Checkpoint(tmp_path / "a", store, {"config": 1}, min_interval=0).close()
    raises(ValueError, Checkpoint, tmp_path / "b", store, {"config": 2})


def test_a_corrupt_step_is_not_done_and_is_cleared(tmp_path):
    checkpoint = Checkpoint(tmp_path / "a", LocalStore(tmp_path / "store"), {}, min_interval=0)
    seal(checkpoint, "s", "good")
    # Nothing touches a sealed step while it uploads (on NFS a file open in the
    # uploader cannot be removed), so wait for the upload before tampering.
    checkpoint.sync()
    (checkpoint.path("s") / "value.txt").write_text("tampered")
    assert not checkpoint.done("s") and not checkpoint.path("s").exists()
    checkpoint.begin("partial")
    assert not checkpoint.done("partial")
    checkpoint.close()


def test_a_later_batch_replaces_a_redone_step(tmp_path):
    store = LocalStore(tmp_path / "store")
    first = Checkpoint(tmp_path / "a", store, {}, min_interval=0)
    directory = first.begin("s")
    (directory / "old.txt").write_text("old")
    first.complete("s")
    first.sync()
    seal(first, "s", "new")
    first.close()
    second = Checkpoint(tmp_path / "b", store, {}, min_interval=0)
    assert second.done("s") and not (second.path("s") / "old.txt").exists()
    second.close()


def test_uploads_are_batched(tmp_path):
    store = LocalStore(tmp_path / "store")
    checkpoint = Checkpoint(tmp_path / "a", store, {}, min_interval=3600)
    for index in range(20):
        seal(checkpoint, f"s{index}", str(index))
    checkpoint.close()
    batches = [name for name in store.names() if name.startswith("batch-")]
    assert 1 <= len(batches) <= 2, batches


# ----------------------------------------------------------------------- pool
# A test worker: study.pool.serve running each job's code.
WORKER = [sys.executable, "-c",
          f"import sys; sys.path.insert(0, {str(HERE)!r}); from study.pool import serve; "
          "serve(lambda spec: exec(spec['code'], {}))"]


def python_job(tmp_path, key, code, threads=1, deps=(), timeout=60.0, priority=0.0):
    return Job(key=key, spec={"code": code}, threads=threads, log=tmp_path / f"{key}.log", deps=tuple(deps),
               priority=priority, timeout=timeout)


def pool(jobs, threads, on_finish=lambda job, outcome: None):
    return run_jobs(jobs, threads, on_finish, lambda budget: WORKER)


def test_pool_keeps_its_thread_budget_and_isolates_failures(tmp_path):
    marker = tmp_path / "running"
    marker.mkdir()
    code = ("import os, time, pathlib; p = pathlib.Path(%r) / str(time.time_ns()); p.write_text("
            "os.environ['RAYON_NUM_THREADS']); time.sleep(0.6); p.unlink()") % str(marker)
    jobs = [python_job(tmp_path, f"j{i}", code, threads=2) for i in range(6)]
    jobs.append(python_job(tmp_path, "bad", "raise ValueError('planted')"))
    peak = []

    def running_threads(job, outcome):
        total = 0
        for path in marker.iterdir():
            try:
                total += int(path.read_text() or 0)
            except FileNotFoundError:
                pass  # that job finished between the listing and the read
        peak.append(total)
    outcomes = pool(jobs, 4, running_threads)
    assert outcomes["bad"].status == "error" and "planted" in (tmp_path / "bad.log").read_text()
    assert all(outcomes[f"j{i}"].status == "ok" for i in range(6))
    assert max(peak) <= 4


def test_workers_are_reused_and_a_crash_replaces_only_its_worker(tmp_path):
    pids = tmp_path / "pids"
    record = f"import os; open({str(pids)!r}, 'a').write(str(os.getpid()) + chr(10))"
    jobs = [python_job(tmp_path, f"a{i}", record, priority=10 - i) for i in range(3)]
    jobs.append(python_job(tmp_path, "crash", "import os; os._exit(3)", priority=5))
    jobs += [python_job(tmp_path, f"b{i}", record, priority=1 - i, deps=["crash"]) for i in range(2)]
    outcomes = pool(jobs, 1)
    assert outcomes["crash"].status == "error" and outcomes["crash"].exit_code == 3
    assert all(outcomes[k].status == "ok" for k in ("a0", "a1", "a2", "b0", "b1"))
    seen = pids.read_text().split()
    # One worker ran the first three jobs; the crash cost it; a new one ran the rest.
    assert len(set(seen[:3])) == 1 and len(set(seen[3:])) == 1 and seen[0] != seen[3]


def test_pool_times_out_restarts_after_a_signal_and_orders_dependencies(tmp_path):
    flag = tmp_path / "once"
    kill_once = (f"import os, signal, pathlib; p = pathlib.Path({str(flag)!r})\n"
                 "if not p.exists(): p.write_text('1'); os.kill(os.getpid(), signal.SIGKILL)")
    order = tmp_path / "order.txt"
    jobs = [python_job(tmp_path, "slow", "import time; time.sleep(30)", timeout=1.0),
            python_job(tmp_path, "killed", kill_once),
            python_job(tmp_path, "first", f"open({str(order)!r}, 'a').write('first ')"),
            python_job(tmp_path, "second", f"open({str(order)!r}, 'a').write('second')", deps=["first"],
                       priority=10)]
    started = time.monotonic()
    outcomes = pool(jobs, 8)
    assert outcomes["slow"].status == "timeout" and time.monotonic() - started < 15
    assert outcomes["killed"].status == "ok" and outcomes["killed"].restarted
    assert order.read_text() == "first second"


def test_an_invalid_artifact_turns_ok_into_error(tmp_path):
    def check(job, outcome):
        if outcome.status == "ok":
            raise ValueError("no model")
    outcomes = pool([python_job(tmp_path, "j", "pass")], 1, check)
    assert outcomes["j"].status == "error"


# --------------------------------------------------------------------- digest
def cell(stratum, n, cases, model="binary", fit="pooled", variant="ours", auc=0.7, horizon=None):
    return {"disease": "t2d", "model": model, "variant": variant, "fit": fit, "stratum": stratum,
            "horizon": horizon, "n": n, "cases": cases, "auc": auc, "observed_risk": cases / n}


def published(rows):
    out = digest.suppress(rows, 20, REGISTRY)
    return {(r["model"], r["fit"], digest.slug(r["stratum"])): r for r in out}


def test_tokens_round_trip():
    rows = [cell("overall", 1000, 100), cell("ancestry:afr", 400, 40), cell("ancestry:eur", 600, 60)]
    names = digest.names(digest.suppress(rows, 20, REGISTRY), 20, REGISTRY)
    parsed, _ = digest.parse(names)
    overall = next(r for r in parsed if r["stratum"] == "overall")
    assert overall["n"] == 1000 and overall["cases"] == 100 and abs(overall["auc"] - 0.7) < 1e-9


def test_small_cells_and_their_complements_are_withheld():
    rows = [cell("overall", 1000, 100), cell("ancestry:afr", 400, 12), cell("ancestry:eur", 600, 88)]
    out = published(rows)
    assert out[("binary", "pooled", "ancestry_afr")] == {**{k: rows[1][k] for k in digest.KEYS},
                                                         "support": digest.INSUFFICIENT}
    eur = out[("binary", "pooled", "ancestry_eur")]
    assert "n" not in eur and "cases" not in eur and "observed_risk" not in eur and eur["auc"] == 0.7
    assert out[("binary", "pooled", "overall")]["cases"] == 100


def test_a_small_remainder_withholds_the_family():
    # 1000 - 400 - 590 = 10 people in an unlisted category.
    out = published([cell("overall", 1000, 100), cell("ancestry:afr", 400, 40), cell("ancestry:eur", 590, 59)])
    assert "n" not in out[("binary", "pooled", "ancestry_eur")]


def test_non_cases_count_as_a_derivable_count():
    out = published([cell("overall", 1000, 990)])
    assert out[("binary", "pooled", "overall")].get("support") == digest.INSUFFICIENT


def test_region_withholding_reaches_divisions():
    rows = [cell("overall", 2000, 200), cell("region:northeast", 1985, 190), cell("region:south", 15, 10),
            cell("division:new_england", 1000, 100), cell("division:middle_atlantic", 985, 90),
            cell("division:south_atlantic", 15, 10)]
    out = published(rows)
    assert "n" not in out[("binary", "pooled", "division_new_england")]


def test_survival_nested_in_binary_is_checked():
    rows = [cell("overall", 1000, 100), cell("overall", 990, 50, model="survival", horizon=1.0)]
    out = published(rows)
    assert "n" not in out[("survival", "pooled", "overall")]


def test_logo_rows_publish_scores_only():
    rows = [cell("overall", 1000, 100), cell("ancestry:afr", 400, 40), cell("ancestry:eur", 600, 60),
            cell("overall", 400, 40, fit="logo:ancestry:afr")]
    logo = published(rows)[("binary", "logo:ancestry:afr", "overall")]
    assert set(logo) - set(digest.KEYS) == {"auc"}


def test_the_last_guard_refuses_a_small_count():
    raises(ValueError, digest.names, [cell("overall", 1000, 15)], 20, REGISTRY)
    raises(ValueError, digest.suppress, [{**cell("overall", 1000, 100), "mystery": 1.0}], 20, REGISTRY)


def test_names_pack_long_cells_and_parse_back():
    row = {**cell("overall", 1000, 100), **{f"d_auc_standard_{i}": 0.001 * i for i in range(60)}}
    registry = digest.Registry({**REGISTRY.specs, **{f"d_auc_standard_{i}": {"type": "score"} for i in range(60)}})
    names = digest.names([row], 20, registry)
    assert len(names) > 1 and all(len(name) <= digest.NAME_BUDGET for name in names)
    parsed, _ = digest.parse(names)
    assert len(parsed) == 1 and parsed[0]["d_auc_standard_59"] == 0.059 and parsed[0]["n"] == 1000


def test_flows_never_step_by_a_small_count():
    flow = {"base": [{"step": "cdr_persons", "n": 5000}, {"step": "in_ancestry", "n": 4990},
                     {"step": "adult", "n": 4500}, {"step": "lookback", "n": 4490}]}
    row, = digest.flow_rows("base", flow, 20)
    # 4990 is 10 from 5000, and 4500 is 10 from the end: neither shows; the ends always do.
    assert row["step_00_cdr_persons"] == 5000 and row["step_03_lookback"] == 4490
    assert not any(name.startswith(("step_01", "step_02")) for name in row)
    # The auditor's chain relation confirms every shown step removes 0 or more than 20.
    parsed, _ = digest.parse(digest.names([row], 20, REGISTRY))
    assert digest.audit([row], REGISTRY) == [] and parsed[0]["step_03_lookback"] == 4490


def test_cohort_counts_by_ancestry_are_partitioned_and_nested():
    by_ancestry = {"afr": {"binary_n": 400, "binary_cases": 60, "survival_n": 390, "survival_disease": 30,
                           "survival_death": 25, "survival_exclusion": 0},
                   "eur": {"binary_n": 900, "binary_cases": 150, "survival_n": 820, "survival_disease": 70,
                           "survival_death": 40, "survival_exclusion": 0}}
    rows = digest.cohort_rows("htn", by_ancestry)
    out = {(r["model"], r["stratum"]): r for r in digest.suppress(rows, 20, REGISTRY)}
    # afr's survival cell sits 10 below its binary cell, so it is withheld, and so its family.
    assert "n" not in out[("cohort_survival", "ancestry_afr")] and "n" not in out[("cohort_survival", "ancestry_eur")]
    assert out[("cohort_binary", "ancestry_afr")]["n"] == 400 and out[("cohort_binary", "overall")]["n"] == 1300
    assert digest.audit(digest.suppress(rows, 20, REGISTRY), REGISTRY, nested={"htn"}) == []


def test_ehr_follow_up_releases_only_safe_fractions():
    row, = digest.ehr_rows({"n": 50000, "fraction_without_ehr": 0.2, "fraction_obs_end_after_ehr_end": 0.9999,
                            "median_gap_years": 0.4, "median_positive_gap_years": 1.1,
                            "fraction_deaths_after_ehr_end": 0.3})
    assert row["with_ehr_n"] == 40000 and row["fraction_without_ehr"] == 0.2
    assert "fraction_obs_end_after_ehr_end" not in row and "fraction_deaths_after_ehr_end" not in row
    assert digest.audit([row], REGISTRY) == []


def test_ehr_domain_fractions_need_their_denominator_and_a_safe_count():
    manifest = {"ehr_people": 300000, "ehr_extended_by": {"procedure": 0.12, "drug": 0.00004},
                "ehr_end_from_long_visit": 0.031}
    row, = digest.ehr_domain_rows(manifest)
    # 0.00004 of 300,000 is 12 people: withheld; the others are safe.
    assert row["extended_by_procedure"] == 0.12 and "extended_by_drug" not in row
    assert row["end_from_long_visit"] == 0.031 and digest.audit([row], REGISTRY) == []
    assert digest.ehr_domain_rows({"ehr_extended_by": {"procedure": 0.12}}) == []


def test_followup_fractions_that_invert_to_small_counts_are_withheld():
    row, = digest.followup_rows({"n": 100000, "fraction_reaching": {"1": 0.9999, "3": 0.61}}, 20)
    assert "reach_h1" not in row and row["reach_h3"] == 0.61


def test_suppressed_names_pass_the_differencing_audit_and_a_planted_leak_fires_it():
    rows = [cell("overall", 1000, 100), cell("ancestry:afr", 400, 12), cell("ancestry:eur", 600, 88)]
    assert digest.audit(digest.suppress(rows, 20, REGISTRY), REGISTRY) == []
    results, _ = digest.encode(rows, [], REGISTRY)
    assert results
    # Planted mutation: primary suppression alone (afr dropped, eur shown) leaves
    # afr = overall - eur derivable, and the audit gate must fire.
    assert digest.audit([rows[0], rows[2]], REGISTRY)
    original = digest.suppress
    try:
        digest.suppress = lambda rows, limit, registry: [rows[0], rows[2]]
        raises(ValueError, digest.encode, rows, [], REGISTRY)
    finally:
        digest.suppress = original


# ----------------------------------------------------------------- provenance
def test_a_planted_pooled_standardization_fires_the_logo_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("study_driver", HERE / "study.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"person_id": np.arange(1000, dtype=np.int64), "pgs": rng.gamma(2, 1, 1000),
                          "test": rng.random(1000) < 0.2,
                          "ancestry": rng.choice(["afr", "eur"], 1000, p=[0.3, 0.7])})
    fit = "logo:ancestry:afr"
    train = frame.loc[driver.training_rows(frame, fit)]
    record = {"train_rows": len(train), "train_sha256": driver.person_set_hash(train.person_id),
              "held_out_in_train": 0, "standardization": driver.standardize(train.pgs)}
    assert driver.verify_provenance(frame, fit, record) == record["standardization"]
    pooled = frame.loc[driver.training_rows(frame, "pooled")]
    planted = dict(record, standardization=driver.standardize(pooled.pgs))
    raises(ValueError, driver.verify_provenance, frame, fit, planted)
    raises(ValueError, driver.verify_provenance, frame, fit, dict(record, train_rows=len(pooled)))


if __name__ == "__main__":
    failures = 0
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
                try:
                    test(Path(directory)) if test.__code__.co_argcount else test()
                    print(f"ok   {name}")
                except Exception as error:
                    failures += 1
                    print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"EXIT rc={failures}")
    sys.exit(1 if failures else 0)
