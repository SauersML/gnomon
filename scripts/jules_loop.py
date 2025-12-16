#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import requests
import re

JULES_API_URL = "https://jules.googleapis.com"

def strip_ansi(text):
    """Removes ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def run_command(cmd, check=False):
    # Support both string (shell=True) and list (shell=False)
    if isinstance(cmd, list):
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    else:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def get_previous_run_info():
    """Retrieves status and logs from the workflow run that triggered this script."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        print("Warning: GITHUB_EVENT_PATH not set. Not triggered by workflow_run?")
        return None, None, None

    with open(event_path) as f:
        event = json.load(f)

    run = event.get("workflow_run")
    if not run:
        print("No workflow_run in event.")
        return None, None, None

    conclusion = run.get("conclusion")
    run_id = run.get("id")
    workflow_id = run.get("workflow_id") # Use ID to re-trigger same workflow

    logs = ""
    print(f"Run {run_id} concluded with: {conclusion}")

    if conclusion != "success":
        print("Fetching logs for failed run...")
        out, err, code = run_command(f"gh run view {run_id} --log")
        if code == 0:
            # Strip ANSI codes to make logs cleaner for the LLM
            logs = strip_ansi(out)
            if len(logs) > 50000:
                print("Truncating logs...")
                logs = logs[-50000:]
        else:
            logs = f"Failed to fetch logs: {err}"
    else:
        logs = "Build passed successfully."

    return conclusion, logs, workflow_id

def call_jules(prompt):
    """Interacts with Jules API to get a plan and changeset."""
    api_key = os.environ.get("JULES_API_KEY")
    repo = os.environ.get("GITHUB_REPOSITORY")

    if not api_key:
        print("Error: JULES_API_KEY not set.")
        sys.exit(1)

    print("Creating Jules session...")
    resp = requests.post(
        f"{JULES_API_URL}/v1alpha/sessions",
        headers={"X-Goog-Api-Key": api_key},
        json={
            "prompt": prompt,
            "sourceContext": {
                "source": f"sources/github/{repo}",
                "githubRepoContext": {"startingBranch": "main"}
            }
        }
    )
    if resp.status_code != 200:
        print(f"Failed to create session: {resp.text}")
        sys.exit(1)

    session = resp.json()
    session_name = session["name"]
    print(f"Session created: {session_name}")

    max_retries = 60 # 10 minutes
    for _ in range(max_retries):
        time.sleep(10)
        print("Polling activities...")

        r = requests.get(
            f"{JULES_API_URL}/v1alpha/{session_name}/activities",
            headers={"X-Goog-Api-Key": api_key}
        )
        if r.status_code != 200:
            print(f"Error polling: {r.text}")
            continue

        activities = r.json().get("activities", [])
        for act in reversed(activities):
            if "artifacts" in act:
                for art in act["artifacts"]:
                    if "changeSet" in art:
                        print("Found ChangeSet!")
                        return art["changeSet"]

    print("Timed out waiting for Jules to produce a ChangeSet.")
    sys.exit(1)

def main():
    conclusion, logs, workflow_id = get_previous_run_info()

    if conclusion == "success":
        prompt = (
            "The previous Lean Proof GHA run passed successfully. "
            "Please find one thing to do, improve, fix, or strengthen in the Lean files "
            "(e.g., lakefile.lean or files in proofs/). "
            "You can optimize code, add comments, strengthen proofs, or refactor. "
            "IMPORTANT: Ensure your changes compile and that all proofs are valid. Do not break existing functionality."
        )
    elif conclusion:
        prompt = (
            f"The previous Lean Proof GHA run failed. "
            f"Here are the logs from the run (ANSI colors stripped):\n\n{logs}\n\n"
            "Please analyze the logs and fix the errors in the Lean files. "
            "IMPORTANT: Ensure your changes compile and that all proofs are valid."
        )
    else:
        prompt = (
            "Please find one thing to improve, fix, or strengthen in the Lean files. "
            "IMPORTANT: Ensure your changes compile and that all proofs are valid."
        )

    print(f"Prompting Jules with: {prompt[:200]}...")

    changeset = call_jules(prompt)

    patch = changeset.get("gitPatch", {}).get("unidiffPatch")
    msg = changeset.get("gitPatch", {}).get("suggestedCommitMessage", "Jules Improvement")

    if not patch:
        print("Jules returned a ChangeSet but no unidiffPatch.")
        sys.exit(0)

    print("Applying patch...")
    with open("jules.patch", "w") as f:
        f.write(patch)

    # Ensure we are on the branch (detached HEAD fix)
    # We fetch origin main and checkout it.
    run_command("git fetch origin main")
    # Force checkout main to match origin/main, resetting any local divergence
    run_command("git checkout -B main origin/main")

    out, err, code = run_command("git apply jules.patch")
    if code != 0:
        print(f"Failed to apply patch: {err}")
        sys.exit(1)

    run_command('git config user.name "Jules Bot"')
    run_command('git config user.email "jules-bot@google.com"')

    run_command("git add .")
    _, _, code = run_command("git diff --cached --quiet")

    if code == 0:
        print("No changes to commit after applying patch.")
        sys.exit(0)

    print("Committing changes...")
    # Use list args to avoid shell injection in commit message
    run_command(['git', 'commit', '-m', msg], check=True)

    print("Pushing changes...")
    run_command("git push origin main", check=True)

    print("Triggering Lean Prover CI...")
    # Use workflow_id if available, else fallback to prover.yml
    target_workflow = workflow_id if workflow_id else "prover.yml"
    run_command(f"gh workflow run {target_workflow}", check=True)
    print("Loop iteration complete.")

if __name__ == "__main__":
    main()
