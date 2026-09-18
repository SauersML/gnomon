"""Run study jobs side by side under one thread budget, in persistent workers.

The gamfit hot loop is partly serial, so a fit given more threads than it can
use wastes them; running many fits at once is what fills a task. A worker is a
process with a fixed thread budget (its native pools size themselves once)
that imports the stack once and runs jobs one after another, so a small fit
costs its fit, not an interpreter start and a cold import from network
storage. A worker that crashes or overruns its job's time is killed and
replaced: the failure ends only that job.

Protocol: the driver writes one JSON job per line to a worker's stdin; the
worker points its stdout and stderr at the job's log, runs it, and writes one
JSON result line ({"status": "ok" | "error", "cpu_seconds", "max_rss_mb"}) to
the file descriptor named by STUDY_RESULT_FD. `serve` is the worker side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import resource
import select
import signal
import subprocess
import sys
import time
import traceback


@dataclass
class Job:
    key: str
    spec: dict
    threads: int
    log: Path
    deps: tuple = ()
    # Larger first: the longest fits start while the budget is empty.
    priority: float = 0.0
    timeout: float = 900.0
    # Jobs with one affinity (study.py: the frame they read) prefer the idle
    # worker that last ran one, so its cached frame is reused.
    affinity: str = ""


@dataclass
class Outcome:
    status: str                 # ok, error, signal, timeout
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    max_rss_mb: float = 0.0     # the worker's peak so far: an upper bound for the job
    exit_code: int | None = None
    restarted: bool = False
    threads: int = 0
    extra: dict = field(default_factory=dict)


def thread_env(threads):
    """Every native pool a worker can start gets the worker's budget."""
    count = str(threads)
    return dict(os.environ, RAYON_NUM_THREADS=count, OMP_NUM_THREADS=count, OPENBLAS_NUM_THREADS=count,
                MKL_NUM_THREADS=count, NUMEXPR_NUM_THREADS=count, POLARS_MAX_THREADS=count)


def signal_group(pid, signum):
    # killpg(1) is kill(-1): every process this account owns. A worker's
    # session id is its pid, which is never 0 or 1.
    if pid <= 1:
        raise ValueError(f"refusing to signal process group {pid}")
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


class Worker:
    def __init__(self, command, threads):
        self.threads = threads
        read, write = os.pipe()
        env = dict(thread_env(threads), STUDY_RESULT_FD=str(write))
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, start_new_session=True, env=env,
                                        pass_fds=(write,), text=True)
        os.close(write)
        self.results = os.fdopen(read, "r")
        self.job = None
        self.started = None
        self.affinity = None

    def start(self, job):
        self.job, self.started, self.affinity = job, time.monotonic(), job.affinity
        self.process.stdin.write(json.dumps({"spec": job.spec, "log": str(job.log)}) + "\n")
        self.process.stdin.flush()

    def kill(self):
        signal_group(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            signal_group(self.process.pid, signal.SIGKILL)
            self.process.wait()
        self.close()

    def close(self):
        for stream in (self.process.stdin, self.results):
            try:
                stream.close()
            except OSError:
                pass


def run_jobs(jobs, total_threads, on_finish, command, progress=None, poll=0.05):
    """Run `jobs` (dependencies first) within `total_threads`; returns key -> Outcome.

    `command(threads)` is the argv of a worker with that thread budget. A job
    starts once every dependency has finished, however it ended (the job itself
    decides what a failed dependency means). A failure is a result, never a
    reason to stop the others. `on_finish(job, outcome)` runs in this thread as
    each job ends and may turn an ok outcome into an error by raising ValueError
    (an invalid artifact). A job whose worker died by a signal it was not sent
    is run once more in a fresh worker.
    """
    keys = [job.key for job in jobs]
    if len(set(keys)) != len(keys):
        raise ValueError("job keys must be unique")
    known = set(keys)
    for job in jobs:
        missing = set(job.deps) - known
        if missing:
            raise ValueError(f"job {job.key} depends on unknown jobs {sorted(missing)}")
    waiting = sorted(jobs, key=lambda job: -job.priority)
    idle, busy, outcomes, restarted = [], [], {}, set()
    used = 0

    def finish(job, outcome):
        if outcome.status == "ok":
            try:
                on_finish(job, outcome)
            except ValueError as error:
                outcome.status = "error"
                outcome.extra["invalid_output"] = type(error).__name__
                on_finish(job, outcome)
        else:
            on_finish(job, outcome)
        outcomes[job.key] = outcome
        if progress is not None:
            progress(len(outcomes), len(jobs))

    def worker_for(job):
        same = [w for w in idle if w.threads == job.threads]
        for worker in sorted(same, key=lambda w: w.affinity != job.affinity):
            idle.remove(worker)
            return worker
        # At most total_threads worker processes: retire an idle one of another budget first.
        while idle and len(idle) + len(busy) >= total_threads:
            idle.pop(0).kill()
        return Worker(command(job.threads), job.threads)

    def terminate(signum, frame):
        raise InterruptedError(f"study pool received signal {signum}")
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        while waiting or busy:
            # Start what fits, backfilling smaller jobs behind a large one.
            still = []
            for job in waiting:
                if all(dep in outcomes for dep in job.deps) and (used + job.threads <= total_threads or not busy):
                    worker = worker_for(job)
                    worker.start(job)
                    busy.append(worker)
                    used += job.threads
                else:
                    still.append(job)
            waiting = still
            if waiting and not busy:
                raise ValueError("study jobs have a dependency cycle")
            ready, _, _ = select.select([w.results for w in busy], [], [], poll)
            now = time.monotonic()
            for worker in list(busy):
                job = worker.job
                line = worker.results.readline() if worker.results in ready else None
                code = worker.process.poll()
                if line:
                    result = json.loads(line)
                    outcome = Outcome(result["status"], seconds=now - worker.started,
                                      cpu_seconds=result["cpu_seconds"], max_rss_mb=result["max_rss_mb"],
                                      restarted=job.key in restarted, threads=job.threads)
                    busy.remove(worker)
                    idle.append(worker)
                elif code is not None or line == "":
                    # The worker died mid-job (a crash, an abort, the OOM killer).
                    code = worker.process.wait()
                    worker.close()
                    busy.remove(worker)
                    if code < 0 and job.key not in restarted:
                        restarted.add(job.key)
                        with Path(job.log).open("a") as handle:
                            handle.write(f"study_pool_restart_after_signal {-code}\n")
                        used -= job.threads
                        waiting.insert(0, job)
                        continue
                    outcome = Outcome("signal" if code < 0 else "error", seconds=now - worker.started,
                                      exit_code=code, restarted=job.key in restarted, threads=job.threads)
                elif now - worker.started > job.timeout:
                    worker.kill()
                    busy.remove(worker)
                    outcome = Outcome("timeout", seconds=now - worker.started, restarted=job.key in restarted,
                                      threads=job.threads)
                else:
                    continue
                used -= job.threads
                finish(job, outcome)
    except BaseException:
        for worker in busy:
            worker.kill()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        for worker in idle:
            worker.kill()
    return outcomes


def serve(run):
    """The worker side: run each job from stdin with `run(spec)`, its output in
    the job's log, and report each outcome on STUDY_RESULT_FD. A Python
    exception fails only its job; anything that kills the process is the
    driver's to see."""
    results = os.fdopen(int(os.environ["STUDY_RESULT_FD"]), "w")
    quiet = os.open(os.devnull, os.O_WRONLY)
    for line in sys.stdin:
        message = json.loads(line)
        before = resource.getrusage(resource.RUSAGE_SELF)
        log = os.open(message["log"], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(log, 1)
        os.dup2(log, 2)
        os.close(log)
        status = "ok"
        try:
            run(message["spec"])
        except Exception:
            traceback.print_exc()
            status = "error"
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(quiet, 1)
        os.dup2(quiet, 2)
        after = resource.getrusage(resource.RUSAGE_SELF)
        results.write(json.dumps({
            "status": status,
            "cpu_seconds": round(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime, 3),
            "max_rss_mb": round(after.ru_maxrss / 1024, 1)}) + "\n")
        results.flush()
