"""The AoU study as one Google Batch job, submitted from inside the workspace.

The workspace's managed Cromwell cannot place a task on a C3D (AMD Genoa) Spot
VM, and the Batch API is reachable only from inside the workspace perimeter (a
laptop call is refused by VPC Service Controls). So `submit_study.py submit`
stages everything from the laptop as before, writes the job below beside the
staged inputs, and prints one command to paste into the workspace's JupyterLab
terminal, where gcloud is the workspace pet service account. Nothing about the
study changes: the task runs the same commands as study.wdl's `analyze`, on the
same image, with the same checkpoint, status and digest prefixes.

The job has one task of three runnables on a shared work directory:
  1. localize: the pinned google-cloud-cli image copies every input object from
     the workspace bucket into the work directory (the runtime image has no gcloud);
  2. analyze: study.wdl's command, with each `~{input}` replaced by its localized path;
  3. logs (always runs): the task log is copied under the checkpoint prefix.
A Spot preemption ends the task with a Batch exit code the job retries; the
study resumes from its checkpoint.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Pinned by digest as in the imputation lane's direct-Batch submitter (2026-09-18).
CLI_IMAGE = ("gcr.io/google.com/cloudsdktool/google-cloud-cli@sha256:"
             "87eaf69da735ab8dfc1c640df1e85a467a4feedf2256d35dad368f970d9cd35f")
REGION = "us-central1"
WORK = "/mnt/disks/work"
# C3D high-CPU shapes: vCPU -> GiB (about 1.97 GiB per vCPU). The task asks for cpu
# vCPUs; the shape is the smallest that holds it, and the memory must fit too.
C3D_HIGHCPU_GIB = {4: 8, 8: 16, 16: 32, 30: 59, 60: 118, 90: 177, 180: 354, 360: 708}
BOOT_DISK_GB = 200
# Batch's own exit codes for a VM lost under the task (preemption, recreation), and
# the study's exit 125 for a signal it could not finish under: all retried, the
# study resuming from its checkpoint.
RETRY_EXIT_CODES = [125, 50001, 50002, 50003, 50004, 50006]
MAX_RETRIES = 5
JOB_ID = re.compile(r"[a-z]([a-z0-9-]{0,61}[a-z0-9])?")


def machine_type(cpu, memory_gb):
    """The C3D high-CPU machine for a task of `cpu` vCPUs and `memory_gb` GiB."""
    for vcpus, gib in sorted(C3D_HIGHCPU_GIB.items()):
        if vcpus >= cpu:
            if memory_gb > gib:
                raise ValueError(f"{memory_gb} GiB does not fit c3d-highcpu-{vcpus} ({gib} GiB); use fewer GiB")
            return f"c3d-highcpu-{vcpus}", gib
    raise ValueError(f"no C3D high-CPU machine has {cpu} vCPUs")


def localized(inputs):
    """{field: local path} for every File input of study.wdl, and the copy plan
    [(uri, local path)] in that order."""
    paths, plan = {}, []

    def one(field, uri, folder):
        local = f"{WORK}/in/{folder}/{uri.rsplit('/', 1)[-1]}"
        paths[field] = local
        plan.append((uri, local))

    for field in ("sources", "config", "wheelhouse_archive", "scorer_archive", "ancestry_predictions",
                  "relatedness_prune"):
        one(field, inputs[field], field)
    paths["score_files"] = []
    for uri in inputs["score_files"]:
        local = f"{WORK}/in/score_files/{uri.rsplit('/', 1)[-1]}"
        paths["score_files"].append(local)
        plan.append((uri, local))
    paths["score_weights"] = []
    for uri in inputs["score_weights"]:
        local = f"{WORK}/in/score_weights/{uri.rsplit('/', 1)[-1]}"
        paths["score_weights"].append(local)
        plan.append((uri, local))
    return paths, plan


def localize_script(plan, project):
    # Everything the localizer prints goes to the task log too: the task's own log is
    # the only log the pet account can read (Cloud Logging is closed to it).
    lines = ["set -euo pipefail", f"rm -rf {WORK}/in {WORK}/task", f"mkdir -p {WORK}/task",
             f"exec > >(tee -a {WORK}/task.log) 2>&1", 'echo "[localize] $(date -u +%FT%TZ) start on $(hostname)"',
             f"id; ls -ld {WORK}; df -h {WORK} | tail -1; gcloud auth list 2>&1 | head -3; gcloud config list 2>&1 | head -4"]
    for uri, local in plan:
        if not uri.startswith("gs://"):
            raise ValueError(f"input {uri} is not a workspace object")
        lines.append(f"mkdir -p {json.dumps(local.rsplit('/', 1)[0])}")
        lines.append("ok=0; for n in 1 2 3; do gcloud storage cp --billing-project "
                     f"{json.dumps(project)} {json.dumps(uri)} {json.dumps(local)} && {{ ok=1; break; }}; "
                     "sleep $((n * 5)); done; [ \"$ok\" = 1 ] || { echo \"[localize] cannot fetch "
                     f"{uri}\"; exit 1; }}")
    lines.append(f'echo "[localize] $(date -u +%FT%TZ) $(find {WORK}/in -type f | wc -l) inputs on $(hostname)"')
    return "\n".join(lines) + "\n"


def analyze_script(wdl_path, inputs, paths, shard=None, keep_store=False):
    """study.wdl's `analyze` command with its WDL placeholders filled in; a shard
    runs study.py on its one disease with --shard (see study.py)."""
    text = Path(wdl_path).read_text()
    start = text.index("  command <<<") + len("  command <<<\n")
    body = text[start:text.index("  >>>", start)]
    body = "\n".join(line[4:] if line.startswith("    ") else line for line in body.splitlines())

    def fill(match):
        expression = match.group(1).strip()
        if "caveats" in expression:
            return " ".join("--caveat " + json.dumps(value) for value in inputs["caveats"])
        if expression.startswith("sep="):
            return " ".join(paths[expression.split()[-1]])
        if expression in paths:
            return paths[expression]
        return str(inputs[expression])

    body = re.sub(r"~\{([^}]*)\}", fill, body)
    if "~{" in body:
        raise ValueError("an unfilled WDL placeholder remains in the task command")
    if body.count("study.py run ") != 1:
        raise ValueError("the task command must run study.py once")
    flags = (f"--shard --diseases {json.dumps(shard)} " if shard else "") + ("--keep-checkpoint " if keep_store else "")
    body = body.replace("study.py run ", "study.py run " + flags)
    return f"set -euo pipefail\ncd {WORK}/task\nexec > >(tee -a {WORK}/task.log) 2>&1\n" + body


def logs_script(checkpoint_uri, project):
    return (f"[ -f {WORK}/task.log ] || exit 0\n"
            f"gcloud storage cp --billing-project {json.dumps(project)} {WORK}/task.log "
            f"{json.dumps(checkpoint_uri.rstrip('/') + '/logs/')}task-a${{BATCH_TASK_RETRY_ATTEMPT:-0}}.$(hostname).log "
            f">/dev/null 2>&1 || echo '[logs] upload failed'\nexit 0\n")


def job(name, wdl_path, inputs, project, service_account, cpu, memory_gb, timeout_minutes, shard=None,
        keep_store=False):
    """The Batch job document for one study run, or for one disease shard of it
    (the schema `gcloud batch jobs submit --config` reads)."""
    if not JOB_ID.fullmatch(name):
        raise ValueError(f"{name} is not a Batch job id")
    machine, _ = machine_type(cpu, memory_gb)
    paths, plan = localized(inputs)
    volumes = [f"{WORK}:{WORK}"]

    def container(image, script):
        return {"container": {"imageUri": image, "entrypoint": "/bin/bash", "volumes": volumes,
                              "commands": ["-c", script]}}

    return {
        "labels": {"submitter": "gnomon-study", "run": name, **({"shard": shard} if shard else {})},
        "taskGroups": [{
            "taskCount": "1", "parallelism": "1", "taskCountPerNode": "1",
            "taskSpec": {
                "computeResource": {"cpuMilli": str(cpu * 1000), "memoryMib": str(memory_gb * 1024)},
                # study.py's own timeout ends the analysis; the job allows its setup and log copy on top.
                "maxRunDuration": f"{(timeout_minutes + 30) * 60}s",
                "maxRetryCount": MAX_RETRIES,
                "lifecyclePolicies": [{"action": "RETRY_TASK", "actionCondition": {"exitCodes": RETRY_EXIT_CODES}}],
                "runnables": [
                    container(CLI_IMAGE, localize_script(plan, project)),
                    container(inputs["runtime_image"], analyze_script(wdl_path, inputs, paths, shard, keep_store)),
                    dict(container(CLI_IMAGE, logs_script(inputs["checkpoint_uri"], project)), alwaysRun=True),
                ],
            },
        }],
        "allocationPolicy": {
            "location": {"allowedLocations": [f"regions/{REGION}"]},
            "instances": [{"policy": {
                "machineType": machine, "provisioningModel": "SPOT",
                "bootDisk": {"type": "hyperdisk-balanced", "sizeGb": str(BOOT_DISK_GB), "image": "batch-debian"},
            }}],
            "network": {"networkInterfaces": [{
                "network": f"projects/{project}/global/networks/network",
                "subnetwork": f"regions/{REGION}/subnetworks/subnetwork", "noExternalIpAddress": True}]},
            "serviceAccount": {"email": service_account,
                               "scopes": ["https://www.googleapis.com/auth/cloud-platform"]},
            "labels": {"submitter": "gnomon-study", "run": name},
        },
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }


def split_command(run, shard_jobs, gather_job, project):
    """The orchestrator command for a split run: submit every shard job, wait for all of
    them to succeed, then submit the whole-study job that gathers them (see aou_batch)."""
    bucket = "/".join(gather_job[1].split("/")[:3])
    lines = ["set -uo pipefail", f"P={json.dumps(project)}; L={REGION}",
             # progress as object names under the run's reply prefix, readable while the command runs
             f"note() {{ echo \"[split] $(date -u +%FT%TZ) $*\"; printf '' > /w/empty; gcloud storage cp /w/empty "
             f"\"{bucket}/orchestrator/out/{run}/p/$(date -u +%H%M%S)__$(echo \"$*\" | tr -c 'A-Za-z0-9=_' '_')\" -q 2>/dev/null || true; }}",
             "submit() { gcloud storage cp \"$2\" /w/$1.json -q && "
             "gcloud batch jobs submit \"$1\" --project $P --location $L --config /w/$1.json --format='value(name,status.state)'; }",
             "state() { gcloud batch jobs describe \"$1\" --project $P --location $L --format='value(status.state)' 2>/dev/null; }"]
    for job_name, uri in shard_jobs:
        lines.append(f"submit {job_name} {uri} || echo \"[split] {job_name} not submitted\"")
    names = " ".join(job_name for job_name, _ in shard_jobs)
    lines += [f"shards=({names})", "while true; do done=0; bad=0; for j in \"${shards[@]}\"; do s=$(state $j); case \"$s\" in "
              "SUCCEEDED) done=$((done+1));; FAILED|DELETION_IN_PROGRESS|CANCELLED) bad=$((bad+1));; esac; done; "
              "note succeeded=$done failed=$bad of=${#shards[@]}; "
              "[ $bad -gt 0 ] && exit 1; [ $done -eq ${#shards[@]} ] && break; sleep 60; done",
              f"submit {gather_job[0]} {gather_job[1]}"]
    return "\n".join(lines) + "\n"


def paste_command(name, job_uri, project):
    """One line for the workspace JupyterLab terminal (gcloud there is the pet service account)."""
    return (f"gcloud storage cp {job_uri} /tmp/{name}.json && gcloud batch jobs submit {name} "
            f"--project {project} --location {REGION} --config /tmp/{name}.json")
