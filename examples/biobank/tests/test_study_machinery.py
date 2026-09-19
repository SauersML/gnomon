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
    # Each job records its thread budget and its [start, end] on the system-wide
    # monotonic clock, in a file nothing deletes (a listing racing an unlink
    # raises ESTALE on NFS); the peak overlap is computed afterwards.
    spans = tmp_path / "spans"
    spans.mkdir()
    code = ("import os, time, pathlib; s = time.monotonic_ns(); time.sleep(0.6); "
            "(pathlib.Path(%r) / f'{os.getpid()}_{s}').write_text("
            "f\"{os.environ['RAYON_NUM_THREADS']} {s} {time.monotonic_ns()}\")") % str(spans)
    jobs = [python_job(tmp_path, f"j{i}", code, threads=2) for i in range(6)]
    jobs.append(python_job(tmp_path, "bad", "raise ValueError('planted')"))
    outcomes = pool(jobs, 4)
    assert outcomes["bad"].status == "error" and "planted" in (tmp_path / "bad.log").read_text()
    assert all(outcomes[f"j{i}"].status == "ok" for i in range(6))
    records = [tuple(map(int, path.read_text().split())) for path in spans.iterdir()]
    assert len(records) == 6 and all(threads == 2 for threads, _, _ in records)
    running, peak = 0, 0
    # At one instant an end sorts before a start, so back-to-back jobs do not overlap.
    for _, _, threads in sorted([(s, 1, t) for t, s, _ in records] + [(e, 0, -t) for t, _, e in records]):
        running += threads
        peak = max(peak, running)
    assert peak == 4  # two 2-thread jobs at once: the budget is used and never exceeded


def test_pool_keeps_its_memory_budget_and_measures_each_job(tmp_path):
    from study.pool import status_bytes, task_memory_bytes
    assert 0 < task_memory_bytes() <= 2**60
    # Six jobs that each hold 150 MiB for 1.5 s: the threads allow all six at once,
    # the budget about two. The first of the class runs alone until it is measured.
    spans = tmp_path / "spans"
    spans.mkdir()
    code = ("import os, time, pathlib; s = time.monotonic_ns(); block = bytearray(150 * 2**20); time.sleep(1.5); "
            "(pathlib.Path(%r) / f'{os.getpid()}_{s}').write_text(f'{s} {time.monotonic_ns()}')") % str(spans)
    jobs = [Job(key=f"m{i}", spec={"code": code}, threads=1, log=tmp_path / f"m{i}.log", memory_class="hold", size=1)
            for i in range(6)]
    stats = {}
    budget = status_bytes("self", "VmRSS") + int(2.5 * 170 * 2**20)
    outcomes = run_jobs(jobs, 6, lambda job, outcome: None, lambda threads: WORKER, memory=budget, stats=stats)
    assert all(outcome.status == "ok" and outcome.max_rss_mb >= 150 for outcome in outcomes.values())
    records = [tuple(map(int, path.read_text().split())) for path in spans.iterdir()]
    running, peak = 0, 0
    for _, delta in sorted([(s, 1) for s, _ in records] + [(e, -1) for _, e in records], key=lambda x: (x[0], x[1])):
        running += delta
        peak = max(peak, running)
    assert peak == 2 and stats["memory_waits"] > 0 and stats["max_bytes"] <= budget + 170 * 2**20, (peak, stats)
    assert not any(outcome.extra.get("over_budget") for outcome in outcomes.values())
    # A job that cannot fit even alone still runs, alone, and is flagged over_budget.
    big = Job(key="big", spec={"code": "block = bytearray(200 * 2**20)"}, threads=1, log=tmp_path / "big.log",
              memory_class="big", size=1)
    alone = run_jobs([big], 6, lambda job, outcome: None, lambda threads: WORKER,
                     memory=status_bytes("self", "VmRSS") + 50 * 2**20)
    assert alone["big"].status == "ok" and alone["big"].extra.get("over_budget") is True


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
    jobs.append(python_job(tmp_path, "spin", "while True: pass", timeout=1.5))
    started = time.monotonic()
    outcomes = pool(jobs, 8)
    assert outcomes["slow"].status == "timeout" and time.monotonic() - started < 15
    # A job killed at its cap is charged the CPU its worker spent on it.
    assert outcomes["spin"].status == "timeout" and outcomes["spin"].cpu_seconds > 0.8
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


def test_long_operation_items_parse_back_and_colliding_keys_are_refused():
    item = "primary_open_angle_glaucoma.survival.covariates.exclusion"
    names = digest.operation_names([{"scope": "fits", "item": item, "status": "ok", "wall_seconds": 12.5}])
    _, operations = digest.parse(names)
    assert operations == [{"scope": "fits", "item": item.replace(".", "_"), "status": "ok", "wall_seconds": 12.5}]
    raises(ValueError, digest.operation_names, [{"scope": "fits", "item": "a.b_c", "status": "ok"},
                                                {"scope": "fits", "item": "a_b.c", "status": "error"}])


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


def test_exclusion_exits_release_as_a_subgroup_and_flag_over_one_percent():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tabulate_study", HERE / "tabulate_study.py")
    tabulate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tabulate)
    steps = [{"step": "base", "n": 50000}, {"step": "no_prevalent", "n": 30000}]

    def released(exits):
        rows = digest.flow_rows("t2d", {"survival": steps}, 20, {"survival": {"exclusion_exits": exits}})
        assert digest.audit(rows, REGISTRY) == []
        return digest.parse(digest.names(rows, 20, REGISTRY))[0]
    parsed = released(600)  # 2.0% of the frame: the dependent-censoring caveat
    assert parsed[0]["exclusion_exits_count"] == 600 and "2.0%" in tabulate.exclusion_caveats(parsed)[0]
    assert tabulate.exclusion_caveats(released(100)) == []  # 0.3%: no caveat
    parsed = released(12)  # a small count: withheld, and the table says it cannot show the share
    assert "exclusion_exits_count" not in parsed[0] and "withheld" in tabulate.exclusion_caveats(parsed)[0]


def test_cohort_counts_by_ancestry_are_partitioned_and_nested():
    by_ancestry = {"afr": {"binary_n": 400, "binary_cases": 60, "survival_n": 390, "survival_disease": 30,
                           "survival_death": 25},
                   "eur": {"binary_n": 900, "binary_cases": 150, "survival_n": 820, "survival_disease": 70,
                           "survival_death": 40}}
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


def test_null_ehr_facts_are_withheld_not_compared():
    # A null fraction or median means its denominator was empty: undefined, never released.
    assert digest.ehr_domain_rows({"ehr_people": 300000, "ehr_extended_by": {"procedure": None},
                                   "ehr_end_from_long_visit": None})[0].keys().isdisjoint(
        {"extended_by_procedure", "end_from_long_visit"})
    row, = digest.ehr_rows({"n": 50000, "fraction_without_ehr": None, "median_gap_years": None})
    assert "fraction_without_ehr" not in row and "median_gap_years" not in row


def test_an_undeclared_fit_error_fails_the_run_and_a_declared_refusal_does_not():
    study = driver()
    config = {"variants": ["ours", "calpred"], "declared_refusals": [
        {"kind": "binary", "variant": "calpred", "phrase": "dense hessian shape mismatch", "issue": "gam#3015",
         "reason": "known"}]}
    study.check_declared_refusals(config)
    for bad in ({"kind": "binary", "variant": "calpred", "phrase": "x", "issue": "3015", "reason": "r"},
                {"kind": "binary", "variant": "shipped", "phrase": "x", "issue": "gam#1", "reason": "r"},
                {"kind": "binary", "variant": "calpred", "phrase": "Dense", "issue": "gam#1", "reason": "r"}):
        raises(ValueError, study.check_declared_refusals, {**config, "declared_refusals": [bad]})
    refused = b"gamfit._rust.GamError: dense Hessian shape mismatch 11x11 vs 10x10\n"
    planted = b"gamfit._rust.GamError: fit_table panicked inside Rust boundary: libcublas unavailable\n"
    assert study.declared_refusal(config, {"kind": "binary", "variant": "calpred"}, refused) == "gam#3015"
    # The same message from another variant, or another message from calpred, is not declared.
    assert study.declared_refusal(config, {"kind": "binary", "variant": "ours"}, refused) is None
    assert study.declared_refusal(config, {"kind": "binary", "variant": "calpred"}, planted) is None
    records = {"fits/a": {"status": "ok"}, "fits/b": {"status": INSUFFICIENT}, "fits/c": {
        "status": "error", "category": "gamerror", "declared_refusal": "gam#3015"}}
    assert study.unexpected_failures(records) == {}
    records["fits/d"] = {"status": "error", "category": "gamerror"}  # one planted GamError
    records["predict/e"] = {"status": "timeout", "category": "unclassified"}
    assert study.unexpected_failures(records) == {"fits/d": "gamerror", "predict/e": "unclassified"}


INSUFFICIENT = "insufficient_events"


def test_an_uncertified_fit_is_never_counted_as_converged():
    certification = driver().certification
    fit = lambda converged="absent", status="ok": {"status": status, "info": {} if converged == "absent"
                                                   else {"converged": converged}}
    assert certification([fit(True), fit(True)]) == "certified"
    assert certification([fit(True), fit(False)]) == "not_certified"  # a planted uncertified component
    assert certification([fit(True), fit()]) == "no_certificate"
    assert certification([fit(False, "error"), fit(status="insufficient_events")]) == "no_fit"
    # A serialized numpy bool arrives as a string: refused, never read as certified.
    raises(ValueError, certification, [fit("False")])


def test_the_gam_commit_is_believed_only_for_the_installed_engine(tmp_path):
    built_from = driver().built_from
    record = tmp_path / "PROVENANCE.json"
    raises(RuntimeError, built_from, {}, record)  # a venv without its build record
    record.write_text(json.dumps({"gam_commit": "7008a74cb5" + "0" * 30,
                                  "extension": {"member": "gamfit/_rust.abi3.so", "engine_sha256": "ab" * 32}}))
    assert built_from({"gamfit/_rust.abi3.so": "ab" * 32}, record).startswith("7008a74cb5")
    # A record of another build (a stale LS fit's engine): refused, never recorded.
    raises(RuntimeError, built_from, {"gamfit/_rust.abi3.so": "cd" * 32}, record)


def test_sex_never_enters_a_fit_of_a_single_sex_disease():
    check = driver().check_single_sex
    diseases = [{"slug": "hypertension", "sex": None}, {"slug": "breast_cancer", "sex": "female"}]
    plans = {"survival": [("shared", "death"), ("ours", "disease")]}
    by_rule = lambda kind, variant, component, disease: ["age", *([] if disease["sex"] else ["sex"]), "PC1"]
    check(diseases, plans, by_rule)
    # A planted design that keeps sex for every disease (the P1a shared death fit).
    planted = lambda kind, variant, component, disease: ["age", "sex", "PC1"] if component == "death" else ["age"]
    try:
        check(diseases, plans, planted)
    except ValueError as error:
        assert "breast_cancer" in str(error) and "death" in str(error), error
    else:
        raise AssertionError("a single-sex disease kept sex in its death fit")


def test_every_slope_is_put_on_the_pooled_z():
    to_pooled_z = driver().to_pooled_z
    # g = a * score: its slope per a z of scale s is a * s. A LOGO fit (s = 2.0), the
    # simulator (s = 0.0003) and the pooled fit (s = 1.5) disagree until rescaled.
    a, pooled_sd = 0.4, 1.5
    for sd in (2.0, 0.0003, pooled_sd):
        assert np.allclose(to_pooled_z(np.full((3, 2), a * sd), sd, pooled_sd), a * pooled_sd, rtol=1e-12)


def test_caveats_are_fixed_labels_never_truncated():
    check = digest.check_caveats
    check(["shipped_arm_absent", "quick_build_multistart_timings_not_production", "x" * 90])
    for bad in ("Shipped", "shipped arm", "shipped__arm", "_shipped", "drop_and_refit", "and_x"):
        raises(ValueError, check, [bad])


def test_a_validation_scope_is_refused_on_aou():
    import argparse
    study = driver().Study
    for scope in ({"kinds": ["survival"], "diseases": None}, {"kinds": None, "diseases": ["hypertension"]}):
        args = argparse.Namespace(config=str(HERE / "study.json"), source="bigquery", input=[], claim=False, **scope)
        try:
            study(args)
        except ValueError as error:
            assert "validation run" in str(error), error  # refused for its scope, not for anything else
        else:
            raise AssertionError("an AoU run accepted a validation scope")


def test_the_run_row_totals_the_bigquery_plan():
    fields = driver().manifest_fields
    bigquery = fields({"source": "bigquery", "tables_sha256": "ab" * 32, "cdr_cutoff": "2024-07-01",
                       "cdr_cutoff_source": "observation_period", "ehr_domains": ["visit", "condition"],
                       "bigquery": {"plan_bytes": {"person": 3 * 10**9, "ehr_procedure": 4 * 10**9},
                                    "bytes_billed": 6 * 10**9}})
    assert bigquery["bigquery_plan_bytes"] == 7 * 10**9 and bigquery["bigquery_bytes_billed"] == 6 * 10**9
    assert bigquery["ehr_domains"] == "visit_condition" and bigquery["cdr_cutoff_source"] == "observation_period"
    simulator = fields({"source": "simulator", "seed": 1000, "simulator": {"scenario": "realistic_independent"}})
    assert "bigquery_plan_bytes" not in simulator and simulator["tables_seed"] == 1000
    assert simulator["tables_scenario"] == "realistic_independent"


def test_pc_scale_rows_are_whole_base_scores_that_pass_the_audit():
    rows = digest.pc_scale_rows({"PC1": 109.2, "PC2": 82.0, "PC6": 33.1})
    parsed, _ = digest.parse(digest.names(rows, 20, REGISTRY))
    assert parsed[0]["sd_pc1"] == 109.2 and parsed[0]["sd_pc6"] == 33.1 and digest.audit(rows, REGISTRY) == []


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


# -------------------------------------------------------------------- reasons
def driver():
    import importlib.util
    spec = importlib.util.spec_from_file_location("study_driver", HERE / "study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_model_setting_needs_a_reason_and_no_length_scale_is_allowed():
    check = driver().check_reasons
    config = {"models": {"binary": {"centers": 12}, "survival": {"death": {"model": "cox", "knots": 4}}},
              "reasons": {"models.binary.centers": "12 matched 24 on simulator accuracy at half the cost",
                          "models.survival.death": "the death hazard needs no slope surface (simulator)"}}
    check(config)
    raises(ValueError, check, {**config, "reasons": {"models.survival.death": "x"}})          # centers unexplained
    raises(ValueError, check, {**config, "reasons": {**config["reasons"], "models.binary.gone": "stale"}})
    raises(ValueError, check, {**config, "reasons": {**config["reasons"], "models.binary.centers": " "}})
    raises(ValueError, check, {"models": {"binary": {"length_scale": 0.5}},
                               "reasons": {"models.binary.length_scale": "planted"}})
    # The shipped arm's pin needs its reason too, and a reason on "models" alone excuses nothing.
    raises(ValueError, check, {**config, "shipped": {"calibrate_sha": "de0bd1df"}})
    check({**config, "shipped": {"calibrate_sha": "de0bd1df"},
           "reasons": {**config["reasons"], "shipped.calibrate_sha": "gnomon calibrate as shipped"}})
    raises(ValueError, check, {**config, "reasons": {"models": "everything"}})


def test_claims_need_the_convergence_rule_and_sized_replicates():
    check = driver().check_claims
    config = json.loads((HERE / "study.json").read_text())
    check(config)
    raises(ValueError, check, config, claim_run=True)                  # no planned scenarios
    planned = json.loads(json.dumps(config))
    planned["claims"]["scenarios"] = ["realistic"]
    raises(ValueError, check, planned, claim_run=True)                 # unsized cells: no fallback to 5
    metrics = list(planned["claims"]["margins"])
    cell = {"sd_dev": 0.001, "delta": 0.002, "source_job": "1330000", "converged_starts": True}
    # R = max(5, ceil((2 * 1.96 * 0.001 / 0.002)^2)) = max(5, ceil(3.84)) = 5.
    planned["claims"]["replicates"]["by_scenario_metric"] = {f"realistic.{m}": {**cell, "R": 5} for m in metrics}
    check(planned, claim_run=True)
    wide = json.loads(json.dumps(planned))
    wide["claims"]["replicates"]["by_scenario_metric"]["realistic.auc_true"]["sd_dev"] = 0.004   # needs R = 62
    raises(ValueError, check, wide, claim_run=True)
    unconverged = json.loads(json.dumps(planned))
    unconverged["claims"]["replicates"]["by_scenario_metric"]["realistic.oe_true"]["converged_starts"] = False
    raises(ValueError, check, unconverged, claim_run=True)
    drift = json.loads(json.dumps(config))
    drift["convergence"]["max_delta_sd"] = 0.05                        # R1 threshold must match the gate's
    raises(ValueError, check, drift)
    brier = json.loads(json.dumps(config))
    brier["claims"]["margins"]["mse_true"] = {"delta": 0.04, "scale": "relative", "reason": "x"}
    raises(ValueError, check, brier)


def test_the_convergence_gate_flags_a_planted_second_start():
    ratio = driver().delta_over_sd
    rng = np.random.default_rng(3)
    risk = rng.uniform(0.02, 0.4, size=(500, 3))
    assert ratio(risk, risk) == 0.0
    assert ratio(risk, risk + 1e-9) < 0.01
    # One person 0.02 away at one horizon is about 0.2 SD here: not converged.
    planted = risk.copy()
    planted[17, 2] += 0.02
    assert ratio(risk, planted) > 0.01
    assert ratio(risk[:, 0], planted[:, 0]) == 0.0      # binary risks are one column


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
