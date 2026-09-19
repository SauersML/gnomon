"""Run study jobs side by side under one thread budget, in persistent workers.

The gamfit hot loop is partly serial, so a fit given more threads than it can
use wastes them; running many fits at once is what fills a task. A worker is a
process with a fixed thread budget (its native pools size themselves once)
that imports the stack once and runs jobs one after another, so a small fit
costs its fit, not an interpreter start and a cold import from network
storage. A worker that crashes or overruns its job's time is killed and
replaced: the failure ends only that job.

Jobs share the task's memory as well as its threads. Each job's peak resident
set is measured (the worker resets its high-water mark per job), and a job
starts only while every worker's resident set now, each busy one counted at
least at its job's estimate, plus this job's estimate fits the budget. A job's
estimate is the largest peak measured in its memory class, scaled up for a
larger frame; a class not yet measured runs one job at a time, estimated at the
largest peak measured in any class.

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
    # Jobs of one memory class peak alike, in proportion to their size (frame rows).
    memory_class: str = ""
    size: int = 0


@dataclass
class Outcome:
    status: str                 # ok, error, signal, timeout
    seconds: float = 0.0
    cpu_seconds: float = 0.0
    max_rss_mb: float = 0.0     # the job's peak (a died job: its worker's, an upper bound)
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


def status_bytes(pid, field):
    """A /proc status field in bytes (VmRSS: resident now; VmHWM: peak since reset)."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise ValueError(f"/proc/{pid}/status has no {field}")


def task_memory_bytes():
    """The memory this task holds: the smallest cgroup limit on its path (a
    Slurm job's --mem, a container's limit), else the host's MemTotal."""
    limits = [meminfo_bytes("MemTotal")]
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        parts = [part for part in path.split("/") if part]
        if hierarchy == "0":
            root, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):
            root, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        for depth in range(len(parts) + 1):
            limit = root.joinpath(*parts[:depth], name)
            if limit.is_file() and limit.read_text().strip() != "max":
                limits.append(int(limit.read_text().strip()))
    return min(limits)


def meminfo_bytes(field):
    """A /proc/meminfo field in bytes."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise ValueError(f"/proc/meminfo has no {field}")


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
        # CPU the worker has reported for its finished jobs, so a job it dies in
        # is charged what the worker used beyond them.
        self.reported_cpu = 0.0
        self.usage = None

    def start(self, job):
        self.job, self.started, self.affinity = job, time.monotonic(), job.affinity
        self.process.stdin.write(json.dumps({"spec": job.spec, "log": str(job.log)}) + "\n")
        self.process.stdin.flush()

    def finished(self, result):
        self.reported_cpu += result["cpu_seconds"]

    def rss(self):
        """Bytes resident now (0 once the process is gone)."""
        try:
            return status_bytes(self.process.pid, "VmRSS")
        except (FileNotFoundError, ProcessLookupError):
            return 0

    def reap(self, block=True):
        """Wait for the worker process with wait4 (never Popen.wait, which drops
        its rusage); returns its exit code, or None if it is still running."""
        pid, status, usage = os.wait4(self.process.pid, 0 if block else os.WNOHANG)
        if pid == 0:
            return None
        self.process.returncode = os.waitstatus_to_exitcode(status)
        self.usage = usage
        return self.process.returncode

    def unreported(self):
        """(cpu seconds, peak MB) of the job the worker was running when it died."""
        if self.usage is None:
            return 0.0, 0.0
        total = self.usage.ru_utime + self.usage.ru_stime
        return round(max(total - self.reported_cpu, 0.0), 3), round(self.usage.ru_maxrss / 1024, 1)

    def kill(self):
        signal_group(self.process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while self.process.returncode is None and self.reap(block=False) is None:
            if time.monotonic() > deadline:
                signal_group(self.process.pid, signal.SIGKILL)
                self.reap()
                break
            time.sleep(0.05)
        self.close()

    def close(self):
        for stream in (self.process.stdin, self.results):
            try:
                stream.close()
            except OSError:
                pass


def run_jobs(jobs, total_threads, on_finish, command, progress=None, poll=0.05, memory=None, stats=None):
    """Run `jobs` (dependencies first) within `total_threads` and `memory`
    bytes (None: threads alone); returns key -> Outcome. `stats`, a dict, gets
    the peak bytes held at once and how often memory, not threads, held a job.

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
    peaks = {}  # memory class -> (largest peak bytes, its job's size)
    stats = {} if stats is None else stats
    stats.update(max_bytes=0, memory_waits=0)

    def estimate(job):
        if job.memory_class in peaks:
            peak, size = peaks[job.memory_class]
            return peak * max(1.0, job.size / size) if size else peak
        return max((peak for peak, _ in peaks.values()), default=0)

    def held():
        """Bytes held now: the driver's and every worker's resident set, a busy
        worker's at least its job's estimate."""
        return (status_bytes("self", "VmRSS") + sum(max(w.rss(), estimate(w.job)) for w in busy)
                + sum(w.rss() for w in idle))

    def measured(job, outcome):
        peak = outcome.max_rss_mb * 2**20
        if peak > peaks.get(job.memory_class, (0, 0))[0]:
            peaks[job.memory_class] = (peak, job.size)

    def finish(job, outcome):
        measured(job, outcome)
        # A job that could not fit even alone ran because it was alone: flagged, so the
        # task size is revisited rather than the overrun passing silently.
        if memory is not None and status_bytes("self", "VmRSS") + outcome.max_rss_mb * 2**20 > memory:
            outcome.extra["over_budget"] = True
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
            holding = held() if memory is not None else 0
            stats["max_bytes"] = max(stats["max_bytes"], holding)
            for job in waiting:
                if all(dep in outcomes for dep in job.deps) and (used + job.threads <= total_threads or not busy):
                    need = estimate(job)
                    if memory is not None and busy:
                        # A class not yet measured runs one job at a time; an idle
                        # worker's cached frame gives way to a job that needs its memory.
                        if job.memory_class not in peaks and any(w.job.memory_class == job.memory_class
                                                                 for w in busy):
                            still.append(job)
                            continue
                        while holding + need > memory and idle:
                            freed = idle.pop(0)
                            holding -= freed.rss()
                            freed.kill()
                        if holding + need > memory:
                            stats["memory_waits"] += 1
                            still.append(job)
                            continue
                    worker = worker_for(job)
                    worker.start(job)
                    busy.append(worker)
                    used += job.threads
                    holding += need
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
                code = worker.reap(block=False) if worker.process.returncode is None else worker.process.returncode
                if line:
                    result = json.loads(line)
                    worker.finished(result)
                    outcome = Outcome(result["status"], seconds=now - worker.started,
                                      cpu_seconds=result["cpu_seconds"], max_rss_mb=result["max_rss_mb"],
                                      restarted=job.key in restarted, threads=job.threads)
                    busy.remove(worker)
                    idle.append(worker)
                elif code is not None or line == "":
                    # The worker died mid-job (a crash, an abort, the OOM killer).
                    code = worker.process.returncode if worker.process.returncode is not None else worker.reap()
                    worker.close()
                    cpu, rss = worker.unreported()
                    busy.remove(worker)
                    if code < 0 and job.key not in restarted:
                        restarted.add(job.key)
                        with Path(job.log).open("a") as handle:
                            handle.write(f"study_pool_restart_after_signal {-code}\n")
                        used -= job.threads
                        waiting.insert(0, job)
                        continue
                    outcome = Outcome("signal" if code < 0 else "error", seconds=now - worker.started, cpu_seconds=cpu,
                                      max_rss_mb=rss, exit_code=code, restarted=job.key in restarted,
                                      threads=job.threads)
                elif now - worker.started > job.timeout:
                    worker.kill()
                    busy.remove(worker)
                    cpu, rss = worker.unreported()
                    outcome = Outcome("timeout", seconds=now - worker.started, cpu_seconds=cpu, max_rss_mb=rss,
                                      restarted=job.key in restarted, threads=job.threads)
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
        # This job's peak resident set starts from the worker's resident set now.
        Path("/proc/self/clear_refs").write_text("5")
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
            "max_rss_mb": round(status_bytes("self", "VmHWM") / 2**20, 1)}) + "\n")
        results.flush()
