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

def fetch_logs(run_id):
    """Fetches logs for a specific workflow run."""
    print(f"Fetching logs for run {run_id}...")
    out, err, code = run_command(f"gh run view {run_id} --log")
    if code == 0:
        # Strip ANSI codes to make logs cleaner for the LLM
        logs = strip_ansi(out)
        print(f"Logs fetched. Length: {len(logs)} characters.")
        if len(logs) > 50000:
            print("Truncating logs to last 50,000 characters...")
            logs = logs[-50000:]
        return logs
    else:
        print(f"Failed to fetch logs: {err}")
        return f"Failed to fetch logs: {err}"

def get_previous_run_info():
    """Retrieves status and logs from the workflow run that triggered this script, or the latest run."""
    print("\n--- Getting Previous Run Info ---")
    event_path = os.environ.get("GITHUB_EVENT_PATH")

    # 1. Try to use the triggering workflow run info
    if event_path:
        print(f"Reading event from {event_path}")
        try:
            with open(event_path) as f:
                event = json.load(f)

            run = event.get("workflow_run")
            if run:
                conclusion = run.get("conclusion")
                run_id = run.get("id")
                workflow_id = run.get("workflow_id") # Use ID to re-trigger same workflow
                print(f"Triggering run {run_id} found in event. Conclusion: {conclusion}")

                logs = ""
                if conclusion != "success":
                    logs = fetch_logs(run_id)
                else:
                    logs = "Build passed successfully."

                return conclusion, logs, workflow_id
            else:
                print("No 'workflow_run' found in event payload.")
        except Exception as e:
            print(f"Warning: Failed to read or parse GITHUB_EVENT_PATH: {e}")

    # 2. Fallback: Fetch latest run of prover.yml via gh
    print("No triggering workflow_run found. Fetching latest 'Lean Prover CI' (prover.yml) run via gh...")

    out, err, code = run_command(["gh", "run", "list", "--workflow", "prover.yml", "--limit", "1", "--json", "conclusion,databaseId,status"])

    if code != 0 or not out:
        print(f"Failed to fetch runs: {err}")
        return None, None, None

    try:
        runs = json.loads(out)
    except json.JSONDecodeError:
        print(f"Failed to decode gh output: {out}")
        return None, None, None

    if not runs:
        print("No runs found for prover.yml")
        return None, None, None

    latest_run = runs[0]
    conclusion = latest_run.get("conclusion")
    run_id = latest_run.get("databaseId")
    # Use filename as workflow_id for re-triggering
    workflow_id = "prover.yml"

    print(f"Latest run {run_id} status: {latest_run.get('status')}, conclusion: {conclusion}")

    logs = ""
    if conclusion != "success":
        logs = fetch_logs(run_id)
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

    print("\n--- Initializing Jules Session ---")

    payload = {
        "prompt": prompt,
        "sourceContext": {
            "source": f"sources/github/{repo}",
            "githubRepoContext": {"startingBranch": "main"}
        }
    }

    # Print the payload being sent (truncate prompt for readability if needed, but user wants verbosity)
    print("Sending payload to Jules API:")
    print(json.dumps(payload, indent=2))

    resp = requests.post(
        f"{JULES_API_URL}/v1alpha/sessions",
        headers={"X-Goog-Api-Key": api_key},
        json=payload
    )
    if resp.status_code != 200:
        print(f"Failed to create session: {resp.text}")
        sys.exit(1)

    session = resp.json()
    session_name = session["name"]
    print(f"Session created: {session_name}")

    max_retries = 60 # 10 minutes
    seen_ids = set()

    for i in range(max_retries):
        time.sleep(10)
        print(f"Polling activities... (Attempt {i+1}/{max_retries})")

        r = requests.get(
            f"{JULES_API_URL}/v1alpha/{session_name}/activities",
            headers={"X-Goog-Api-Key": api_key}
        )
        if r.status_code != 200:
            print(f"Error polling: {r.text}")
            continue

        activities = r.json().get("activities", [])
        # Sort chronologically to print in order
        activities.sort(key=lambda x: x.get("createTime", ""))

        latest_changeset = None

        for act in activities:
            act_id = act.get("id")
            if act_id in seen_ids:
                continue
            seen_ids.add(act_id)

            originator = act.get("originator", "UNKNOWN")

            print(f"\n--- New Activity ({originator}) ---")

            if "planGenerated" in act:
                print("Plan Generated:")
                steps = act["planGenerated"].get("plan", {}).get("steps", [])
                for step in steps:
                    print(f"  {step.get('index', '?')}. {step.get('title', '')}")

            if "progressUpdated" in act:
                prog = act["progressUpdated"]
                print(f"Status: {prog.get('title', '')}")
                if "description" in prog:
                    print(f"Details: {prog['description']}")

            if "artifacts" in act:
                for art in act["artifacts"]:
                    if "bashOutput" in art:
                        bo = art["bashOutput"]
                        print(f"Bash Command: {bo.get('command')}")
                        print(f"Output:\n{bo.get('output')}")
                    if "changeSet" in art:
                        print("Artifact: ChangeSet found.")
                        latest_changeset = art["changeSet"]
                    if "pullRequest" in art:
                        pr = art["pullRequest"]
                        print(f"Pull Request: {pr.get('title')} - {pr.get('url')}")

            if "sessionCompleted" in act:
                print("Session Completed.")

        if latest_changeset:
            return latest_changeset

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

    print(f"\nPrompting Jules with:\n{prompt}\n")

    changeset = call_jules(prompt)

    patch = changeset.get("gitPatch", {}).get("unidiffPatch")
    msg = changeset.get("gitPatch", {}).get("suggestedCommitMessage", "Jules Improvement")

    if not patch:
        print("Jules returned a ChangeSet but no unidiffPatch.")
        sys.exit(0)

    print("\n--- Applying Patch ---")
    print(f"Patch content:\n{patch}\n")

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

    print("\n--- Committing and Pushing ---")
    print(f"Commit message: {msg}")

    # Use list args to avoid shell injection in commit message
    run_command(['git', 'commit', '-m', msg], check=True)

    print("Pulling latest changes to avoid non-fast-forward...")
    run_command("git pull --rebase origin main", check=True)

    print("Pushing changes...")
    run_command("git push origin main", check=True)

    print(f"\n--- Triggering Next Workflow ({workflow_id}) ---")
    # Use workflow_id if available, else fallback to prover.yml
    target_workflow = workflow_id if workflow_id else "prover.yml"
    run_command(f"gh workflow run {target_workflow}", check=True)
    print("Loop iteration complete.")

if __name__ == "__main__":
    main()
